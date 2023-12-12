def get_random_forest_features(filename):
    # Imports
    from sklearn.ensemble import RandomForestClassifier

    # PART 1: PREPARE DATASET

    # Load dataframe
    vball = pd.read_csv("Volleyball.csv", header=3)

    # Format target column "SET_REGION" and drop rows with missing targets
    def rename_targets(row):
        """Convert target values to 0 (left), 1 (middle), or 2 (right)"""
        region = row["SET_REGION"]

        if region == "FL":
            return 0

        elif region in ["FM", "BM"]:
            return 1

        elif region in ["FR", "BR"]:
            return 2

        else:
            return np.nan

    vball["SET_REGION"] = vball.apply(rename_targets, axis=1)
    vball = vball[vball["SET_REGION"].notna()]

    # Add columns that store location of last few sets
    cur_opp = None
    num_blank = None
    SET_REGION_1_BEFORE, SET_REGION_2_BEFORE, SET_REGION_3_BEFORE = [], [], []
    prev_ind = [None, None, None]

    for i in vball.index:

        if vball.loc[i]["Opponent"] != cur_opp:
            cur_opp = vball.loc[i]["Opponent"]
            num_blank = 3
        else:
            if num_blank > 0:
                num_blank -= 1

        SET_REGION_1_BEFORE.append(vball.loc[prev_ind[-1]]["SET_REGION"] if num_blank < 3 else -1)
        SET_REGION_2_BEFORE.append(vball.loc[prev_ind[-2]]["SET_REGION"] if num_blank < 2 else -1)
        SET_REGION_3_BEFORE.append(vball.loc[prev_ind[-3]]["SET_REGION"] if num_blank < 1 else -1)

        prev_ind[-3] = prev_ind[-2]
        prev_ind[-2] = prev_ind[-1]
        prev_ind[-1] = i

    vball["SET_REGION_1_BEFORE"] = SET_REGION_1_BEFORE
    vball["SET_REGION_2_BEFORE"] = SET_REGION_2_BEFORE
    vball["SET_REGION_3_BEFORE"] = SET_REGION_3_BEFORE

    # Add Boolean columns that measure relative heights of hitters and blockers
    def compare_left(row):
        return row["BYU_FL_Height"] < row["OPP_FL_Height"]
    vball["LeftBlockerTaller"] = vball.apply(compare_left, axis=1)

    def compare_middle(row):
        return row["BYU_FM_Height"] < row["OPP_FM_Height"]
    vball["MiddleBlockerTaller"] = vball.apply(compare_middle, axis=1)

    def compare_right(row):
        return row["BYU_FR_Height"] < row["OPP_FR_Height"]
    vball["RightBlockerTaller"] = vball.apply(compare_right, axis=1)

    # Drop unwanted columns
    # Dropped Columns: Opponent, ALL BYU and Opponent Numbers/IDs, NUM_SETS for specific positions, all specific hitting stats
    some_columns_to_drop = ["Opponent", "BYU_FL_Number", "BYU_FM_Number", "BYU_FR_Number", "BYU_BR_Number", "BYU_BM_Number", "BYU_BL_Number", "OPP_FL_number", "OPP_FL_ID", "OPP_FM_number", "OPP_FM_ID", "OPP_FR_number", "OPP_FR_ID", "OPP_BR_number", "OPP_BR_ID", "OPP_BM_number", "OPP_BM_ID", "OPP_BL_number", "OPP_BL_ID", "NUM_SETS_FL", "NUM_SETS_FM", "NUM_SETS_FR", "NUM_SETS_BR", "NUM_SETS_BM", "NUM_SETS_BL"]
    columns_to_drop = some_columns_to_drop + [col for col in vball.columns if col[0].isnumeric()]
    vball = vball.drop(labels=columns_to_drop, axis=1)

    # Shuffle rows in dataset
    vball = vball.sample(frac=1)

    # 1-Hot Encode Categorical variables
    vball = pd.get_dummies(vball, drop_first=True, columns=["BYU_FL_Position", "BYU_FM_Position", "BYU_FR_Position", "BYU_BR_Position", "BYU_BM_Position", "BYU_BL_Position", "OPP_FL_Position", "OPP_FM_Position", "OPP_FR_Position", "OPP_BR_Position", "OPP_BM_Position", "OPP_BL_Position", "OPP_BLOCKED_BYU_LAST_POINT", "SET_POINT", "IS_POINT_AFTER_TIMEOUT_OR_FIRST_POINT_OF_SET", "SET_REGION_1_BEFORE", "SET_REGION_2_BEFORE", "SET_REGION_3_BEFORE", "LeftBlockerTaller", "MiddleBlockerTaller", "RightBlockerTaller"])

    # Split into features and targets
    X = vball.drop(labels=["SET_REGION"], axis=1)
    # y = vball["SET_REGION"]

    # Part 2: Feature Selection (for the ensemble, I just hard-code the best performing subset so far)
    subset = ['BYU_FL_Season_kills', 'BYU_FM_Attmepts', 'BYU_FR_Height',
     'BYU_FR_Season_errors', 'BYU_BL_Attmepts', 'OPP_FL_SA', 'OPP_FM_BA',
     'OPP_FM_SE', 'OPP_FR_SE', 'OPP_BR_BS', 'OPP_BR_BA', 'OPP_BM_SE', 'OPP_BL_BS',
     'Num_sets_left', 'Num_sets_middle', 'Num_sets_right', 'NUM_CONSEC_SERVES',
     'BYU_SCORE', 'OPP_SCORE', 'SET_REGION_2_BEFORE_0.0',
     'SET_REGION_3_BEFORE_2.0']
    X_reduced = X.loc[:, subset]

    return X_reduced
