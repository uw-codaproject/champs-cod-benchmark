from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    from typing import Annotated, TypedDict
except ImportError:
    from typing_extensions import Annotated, TypedDict

from champs_pipeline.config import ModelConfig, PromptConfig
from champs_pipeline.model_runner import ModelRunner, ModelRunnerPool

try:
    from langgraph.graph import StateGraph, START, END
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "langgraph is required for multi-agent mode. Please install langgraph."
    ) from exc

import json
import queue
import torch


@dataclass
class AgentSpec:
    name: str
    system_prompt: str


class AgentState(TypedDict, total=False):
    user_payload: str
    allowed_causes: List[str]
    specialist_outputs: Annotated[Dict[str, str], operator.or_]
    prompts: Annotated[Dict[str, str], operator.or_]
    top_k_causes: List[str]
    decision: str


class MultiAgentOrchestrator:
    """
    LangGraph-based multi-agent classifier with two strategies:
    - disease (default): One specialist agent per allowed cause (plus OTHERS fallback).
      Uses top-k selection and a decision agent synthesizing specialist outputs.
    - role: Role-based panelists with a panel chair decision agent; no top-k selection.
    """

    def __init__(self, model_cfg: ModelConfig, prompt_cfg: PromptConfig, allowed_causes: List[str]):
        self.model_cfg = model_cfg
        self.base_prompt_cfg = prompt_cfg
        self.allowed_causes = allowed_causes
        self.agent_strategy = getattr(prompt_cfg, "agent_strategy", "disease")
        if self.agent_strategy == "role":
            self.specialists = self._build_role_panelists(prompt_cfg, allowed_causes)
            self.top_k = 0
        else:
            self.specialists = self._build_specialists(prompt_cfg, allowed_causes)
            self.top_k = 3
        self.decision_agent = self._build_decision_agent(prompt_cfg)
        self.graph = self._build_graph()

        # Initialize and load runners once via the pool.
        self.pool = ModelRunnerPool(model_cfg, prompt_cfg)
        self.pool.load()

    def classify(self, user_payload: str) -> Tuple[str, Dict[str, str]]:
        """Run the specialist + decision graph for one case."""
        initial_state: AgentState = {
            "user_payload": user_payload,
            "allowed_causes": self.allowed_causes,
            "specialist_outputs": {},
            "prompts": {},
            "top_k_causes": [],
        }
        final_state: AgentState = self.graph.invoke(initial_state)
        decision = final_state.get("decision", "")
        prompts = final_state.get("prompts", {})
        return decision, prompts

    def classify_batch(self, user_payloads: List[str]) -> List[Tuple[str, Dict[str, str]]]:
        """
        Run the specialist + decision graph for multiple cases in parallel.
        Uses LangGraph's batch execution.
        """
        initial_states: List[AgentState] = [
            {
                "user_payload": payload,
                "allowed_causes": self.allowed_causes,
                "specialist_outputs": {},
                "prompts": {},
                "top_k_causes": [],
            }
            for payload in user_payloads
        ]
        final_states: List[AgentState] = self.graph.batch(initial_states)
        return [
            (s.get("decision", ""), s.get("prompts", {})) 
            for s in final_states
        ]

    def visualize(self) -> str:
        """
        Render the LangGraph as Mermaid (string). Suitable for piping to the CLI.
        """
        if not hasattr(self.graph, "get_graph"):
            return ""
        return self.graph.get_graph().draw_mermaid()

    def save_markdown(self, path: str | Path) -> None:
        """
        Write a Markdown file containing the Mermaid diagram.
        """
        mermaid = self.visualize()
        if not mermaid:
            return
        Path(path).write_text(f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")

    def _build_graph(self):
        graph = StateGraph(AgentState)
        if self.agent_strategy == "disease":
            graph.add_node("run_specialists", self._run_all_specialists)
            graph.add_edge(START, "run_specialists")
            graph.add_node("select_top_k", self._select_top_k)
            graph.add_node("decision", self._run_decision)
            graph.add_edge("run_specialists", "select_top_k")
            graph.add_edge("select_top_k", "decision")
            graph.add_edge("decision", END)
        elif self.agent_strategy == "role":
            graph.add_node("run_specialists", self._run_all_specialists)
            graph.add_edge(START, "run_specialists")
            graph.add_node("decision", self._run_decision)
            graph.add_edge("run_specialists", "decision")
            graph.add_edge("decision", END)
        return graph.compile()

    def _specialist_node(self, spec: AgentSpec):
        def run(state: AgentState) -> AgentState:
            runner = self.pool.get()
            try:
                gen, pr = runner.predict_prompt(state["user_payload"], system_prompt=spec.system_prompt)
                # Only emit changed pieces to avoid concurrent writes to other keys.
                return {
                    "specialist_outputs": {spec.name: gen},
                    "prompts": {spec.name: pr},
                }
            finally:
                self.pool.put(runner)

        return run

    def _run_all_specialists(self, state: AgentState) -> AgentState:
        """
        Execute all specialists for this case. To maximize efficiency, we batch 
        them and run on a single borrowed runner from the pool.
        """
        runner = self.pool.get()
        try:
            specialist_prompts = [state["user_payload"]] * len(self.specialists)
            specialist_sys_prompts = [spec.system_prompt for spec in self.specialists]
            
            gens, ptxts = runner.predict_batches(
                specialist_prompts, system_prompts=specialist_sys_prompts
            )
            
            specialist_outputs = {
                spec.name: gen for spec, gen in zip(self.specialists, gens)
            }
            prompts_dict = {
                spec.name: pr for spec, pr in zip(self.specialists, ptxts)
            }
            return {
                "specialist_outputs": specialist_outputs,
                "prompts": prompts_dict,
            }
        finally:
            self.pool.put(runner)

    def _select_top_k(self, state: AgentState) -> AgentState:
        if self.top_k <= 0:
            return state
        outputs = state.get("specialist_outputs", {})
        top_k_causes = _pick_top_k_causes(outputs, self.allowed_causes, self.top_k)
        return {**state, "top_k_causes": top_k_causes}

    def _run_decision(self, state: AgentState) -> AgentState:
        if self.agent_strategy == "disease":
            top_k_causes = state.get("top_k_causes", []) or self.allowed_causes
            specialist_outputs = state.get("specialist_outputs", {})
            prompts = state.get("prompts", {})

            decision_system_prompt = (self.decision_agent.system_prompt or "").replace(
                "{allowed_causes}", ", ".join(top_k_causes)
            )
            ds_appendix = _extract_ds_sections(self.base_prompt_cfg.diagnosis_standard or "", top_k_causes)
            decisions_text = "\n".join(
                f"- {k}: {v}" for k, v in specialist_outputs.items() if k in top_k_causes
            )
            decision_prompt = f"## SPECIALIST OPINIONS:\n{decisions_text}\n\n{ds_appendix}\n\n{state['user_payload']}"
            
            runner = self.pool.get()
            try:
                final_pred, decision_prompt_text = runner.predict_prompt(decision_prompt, system_prompt=decision_system_prompt)
            finally:
                self.pool.put(runner)
                
            prompts["decision"] = decision_prompt_text
            return {**state, "decision": final_pred, "prompts": prompts}
        elif self.agent_strategy == "role":
            specialist_outputs = state.get("specialist_outputs", {})
            prompts = state.get("prompts", {})

            decision_system_prompt = (self.decision_agent.system_prompt or "").replace(
                "{allowed_causes}", ", ".join(self.allowed_causes)
            )
            ds_appendix = _extract_ds_sections(self.base_prompt_cfg.diagnosis_standard or "", self.allowed_causes)
            decisions_text = "\n".join(
                f"- {k}: {v}" for k, v in specialist_outputs.items()
            )
            decision_prompt = f"## PANELIST OPINIONS:\n{decisions_text}\n\n{ds_appendix}\n\n{state['user_payload']}"
            
            runner = self.pool.get()
            try:
                final_pred, decision_prompt_text = runner.predict_prompt(decision_prompt, system_prompt=decision_system_prompt)
            finally:
                self.pool.put(runner)
                
            prompts["decision"] = decision_prompt_text
            return {**state, "decision": final_pred, "prompts": prompts}

    def _make_runner(self, system_prompt: str) -> ModelRunner:
        cfg_prompt = deepcopy(self.base_prompt_cfg)
        cfg_prompt.system_prompt = system_prompt
        return ModelRunner(self.model_cfg, cfg_prompt)

    def _build_specialists(self, prompt_cfg: PromptConfig, causes: List[str]) -> List[AgentSpec]:
        specs: List[AgentSpec] = []
        base_specialist = getattr(prompt_cfg, "disease_specialist_prompt", None) or prompt_cfg.specialist_prompt or ""
        for cause in causes:
            ds_prompt = _extract_ds_sections(prompt_cfg.diagnosis_standard or "", [cause])
            system_prompt = f"{base_specialist.replace('{cause}', cause).strip()}\n\n{ds_prompt}".strip()
            specs.append(AgentSpec(name=cause, system_prompt=system_prompt))
        return specs

    def _build_role_panelists(self, prompt_cfg: PromptConfig, causes: List[str]) -> List[AgentSpec]:
        base_role_prompt = getattr(prompt_cfg, "role_specialist_prompt", None) or prompt_cfg.specialist_prompt or ""
        roles = getattr(prompt_cfg, "role_agents", None)
        if not roles:
            roles = [{"name": "Role Panelist", "description": ""}]
        ds_appendix = _extract_ds_sections(prompt_cfg.diagnosis_standard or "", causes)
        specs: List[AgentSpec] = []
        for role in roles:
            role_name = role.get("name", "Role Panelist")
            role_desc = role.get("description", "")
            system_prompt = base_role_prompt
            system_prompt = system_prompt.replace("{persona_name}", role_name)
            system_prompt = system_prompt.replace("{persona_description}", role_desc)
            system_prompt = system_prompt.replace("{allowed_causes}", ", ".join(causes))
            if ds_appendix:
                system_prompt = f"{system_prompt.strip()}\n\n{ds_appendix}".strip()
            specs.append(AgentSpec(name=role_name, system_prompt=system_prompt))
        return specs

    def _build_decision_agent(self, prompt_cfg: PromptConfig) -> AgentSpec:
        if self.agent_strategy == "role":
            system_prompt = getattr(prompt_cfg, "panel_chair_prompt", None) or prompt_cfg.decision_prompt
        else:
            system_prompt = getattr(prompt_cfg, "disease_decision_prompt", None) or prompt_cfg.decision_prompt
        return AgentSpec(name="decision", system_prompt=system_prompt)


def _extract_ds_sections(ds_text: str, causes: List[str]) -> str:
    """
    Extract Level of Certainty + specified cause sections from the diagnosis standard text.
    """
    if not ds_text:
        return ""
        
    lines = ds_text.splitlines()

    targets = {c.strip().lower() for c in causes}
    captured: List[str] = ["## DIAGNOSIS STANDARD"]
    keep_section = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if keep_section:
                captured.append(line)
            continue
        if stripped.startswith("## "):
            # If there's another high-level header, stop current section unless it's the main header
            if stripped != "## DIAGNOSIS STANDARD":
                keep_section = None
            continue
        if stripped.startswith("### "):
            heading = stripped[4:].strip().lower()
            if heading == "level of certainty":
                keep_section = "levels"
                captured.append(line)
                continue
            if heading in targets:
                keep_section = "cause"
                captured.append(line)
                continue
            keep_section = None
            continue
        if keep_section in {"levels", "cause"}:
            captured.append(line)

    if len(captured) == 1:
        # If nothing was captured except the header, check if we should just return the whole thing
        # but usually we want to be specific. For now, return empty if no specific sections matched.
        return ""
    return "\n".join(captured).strip()


def _pick_top_k_causes(outputs: Dict[str, str], allowed: List[str], k: int) -> List[str]:
    scored = []
    for cause in allowed:
        text = outputs.get(cause, "") or ""
        score = _extract_confidence(text)
        scored.append((score, cause))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored][:k]


def _extract_confidence(text: str) -> float:
    """
    Very lightweight JSON probability extractor; raises an error if the probability is not a float.
    """
    try:
        obj = json.loads(text)
        val = obj.get("probability", None)
        return float(val)
    except json.JSONDecodeError:
        print(f"Error extracting confidence from: {text}")
        return 0.5
