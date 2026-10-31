CASE_ID_COLUMN = "champs_deid"

TARGET_COLUMNS = [
    "UC_champs_group_desc",
    "IC_champs_group_desc",
    "Morbid_Cond_01_champs_group_desc",
    "Morbid_Cond_02_champs_group_desc",
    "Morbid_Cond_03_champs_group_desc",
    "Morbid_Cond_04_champs_group_desc",
    "Morbid_Cond_05_champs_group_desc",
    "Morbid_Cond_06_champs_group_desc",
    "Morbid_Cond_07_champs_group_desc",
    "Morbid_Cond_08_champs_group_desc",
]

MAIN_TARGET = "UC_champs_group_desc"

DEMO_COLUMNS = [
    "sex", "age_years", "age_months", "age_days",
    "age_hours", "age_group", "age_group_subcat",
    "location_of_death", "hosp_LOS_24h", "hosp_LOS_48h",
    "hosp_LOS_gt_48h", "hosp_LOS_lt_72h", "hosp_LOS_gt_72h",
    "weight_kg", "height_cm", "MUAC_cm", "head_circumf_cm",
    "right_leg_length_cm", "right_foot_length_cm", "GA_maternal_wks",
    "GA_maternal_months", "GA_child_wks", "GA_child_method",
    "BW_maternal", "BW_child", "calc_postmortem_hrs"
]

# Columns to drop by name
VA_COLUMNS_TO_DROP = [
    'Version', 'id10476_deid',
    'Id10481', 'Id10436', 'Id10011', 'Id10019'
]

# Excel range to drop
VA_EXCEL_RANGE_TO_DROP = ('B', 'AD')

## Clinical abstraction stuff

CHILD_ABS_V1_UNKNOWN_VALUES = {"unknown", "nao disponível", "", None, "na"}

CHILD_ABS_V1_COLUMNS_TO_DROP = [
    "cpex_040", "cpex_072", "cpex_084", "cpex_145", "cpex_156",
    "cpex_158", "cpex_159", "cpex_160", "cpex_161", "cpex_178",
    "cpex_188", "cpmh_041", "cpmh_047", "cpmh_050", "cpmh_051",
    "cpmh_053", "cpmh_055", "cpmh_059", "cpmh_060", "cpmh_061",
    "cpex_085",
    # Columns with dates/times that are mostly blank and hard to impute meaningfully
    "cpmh_118", "hvtb_023", "hvtb_031", "hvtb_034", "hvtb_036",
    "hvtb_039", "hvtb_045", "hvtb_057", "hvtb_095", "hvtb_096",
    "hvtb_102", "clab_019", "clab_020", "clab_030", "clab_031",
    # Columns that are hard to clean/encode and mostly blank/definition not clear
    "clab_071", "clab_079", "clab_085", "clab_095", "clab_077",
    "clab_089" # definition doesnt match with values
]

CHILD_ABS_V1_CLINICAL_NARRATIVE_COLUMNS = [
    "cpil_002", "cpil_003", "cpil_057", "cpil_059",
    "cpex_025", "cpex_038", "cpex_083", "cpex_088",
    "cpex_097", "cpex_105", "cpex_130", "cpex_175",
    "cpex_194", "cpmh_003", "cpmh_004", "cpmh_005",
    "cpmh_019", "cpmh_023", "cpmh_025", "cpmh_029",
    "cpmh_072", "cpmh_083", "cpmh_084",
    "cpmh_092", "cpmh_095", "cpmh_143",
    "cpmh_148", "hvtb_085", "hvtb_092", "hvtb_156",
    "hvtb_157", "hvtb_196", "chpc_001", "chpc_033",
    "chpc_047", "clab_006", "clab_068", "clab_069",
    "clab_070", "clab_073", "clab_075", "clab_090",
    "clab_111", "csum_001", "csum_002", "csum_011"
]

CHILD_ABS_V1_TRAIN_ALL_NULL_COLUMNS_INFANTCHILD = [
    "cpmh_065","cpmh_067","cpmh_073","cpmh_080","cpmh_093","cpmh_136",
    "cpmh_138","cpmh_139","cpmh_140","cpmh_151","cpmh_153","cpmh_154",
    "hvtb_058","hvtb_060","hvtb_062","hvtb_064","hvtb_066","hvtb_068",
    "hvtb_070","hvtb_072","hvtb_073","hvtb_074","hvtb_075","hvtb_076",
    "hvtb_077","hvtb_078","hvtb_079","hvtb_081","hvtb_082","hvtb_083",
    "hvtb_087","hvtb_089","hvtb_101","hvtb_108","hvtb_110","hvtb_111",
    "hvtb_122","hvtb_125","hvtb_126","hvtb_127","hvtb_128","hvtb_129",
    "hvtb_130","hvtb_131","hvtb_132","hvtb_133","hvtb_134","hvtb_135",
    "hvtb_136","hvtb_137","hvtb_139","hvtb_144","hvtb_145","hvtb_146",
    "hvtb_147","hvtb_148","hvtb_150","hvtb_152","hvtb_153","hvtb_162",
    "hvtb_165","hvtb_166","hvtb_167","hvtb_168","hvtb_169","hvtb_170",
    "hvtb_171","hvtb_172","hvtb_173","hvtb_174","hvtb_175","hvtb_176",
    "hvtb_177","hvtb_179","hvtb_182","hvtb_183","hvtb_184","hvtb_185",
    "hvtb_187","hvtb_188","hvtb_189","hvtb_190","hvtb_191","hvtb_192",
    "hvtb_193","chpc_020","chpc_021","chpc_022","chpc_025","chpc_027",
    "chpc_028","chpc_029","chpc_032","clab_003","clab_035","clab_039",
    "clab_062","clab_081","clab_104"
]