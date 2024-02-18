import pandas as pd
from IPython.core.display import display, HTML


RESULTS_STYLE = [
    dict(selector="th", props=[("width", "5px"),
                               ("text-align", "left")]),
    dict(
        selector="td.row_heading.level0",
        props=[
            ("font-size", "10pt")],
    ),
    dict(
        selector="th.col_heading.level1",
        props=[
            ("writing-mode", "vertical-rl"),
            ("transform", "rotateZ(180deg)"),
            ("height", "150px"),
            ("vertical-align", "bottom"),
        ],
    ),
]


def _find_dict_values_with_parent(d, parent_key=None, target_keys=["classification"]):
    """Recursively finds values of specified keys in a nested dictionary and
    returns them with their parent key."""
    if isinstance(d, dict):
        for k, v in d.items():
            if (
                k in target_keys
            ):  # Check if the current key is in the list of target keys
                yield (parent_key, v)
            elif isinstance(v, dict):
                yield from _find_dict_values_with_parent(v, k, target_keys)
            elif isinstance(v, list):
                for item in v:
                    yield from _find_dict_values_with_parent(item, k, target_keys)


def _drop_dict_values(result: dict, exclude_keys=["explanation", "classification"]):
    """Helper function to drop/exclude specific keys from a dictionary"""
    return {k: result[k] for k in set(list(result.keys())) - set(exclude_keys)}


def _bool_to_dot(value):
    """Helper function for pandas - convert booleans to dots"""
    if isinstance(value, bool):
        return "●"  # Substitute booleans with dots
    else:
        return value


def _tuple_to_dict(tuple_pairs):
    # Create a structured dictionary to accommodate values for each index
    structured_data = {}
    for key, value in tuple_pairs:
        if key not in structured_data:
            structured_data[key] = [value]
        else:
            structured_data[key].append(value)

    return structured_data


def create_tables_single_result(result):
    tuple_pairs = list(
        _find_dict_values_with_parent(
            result, target_keys=["classification", "explanation", "rating"]
        )
    )
    explanations = _tuple_to_dict(list(tuple_pairs))
    mod_tags = result["moderator"]["mod_results"]["categories"]
    classifier_tags = _drop_dict_values(result["classifier"])
    rw_indicator = _drop_dict_values(result["right_wing_rater"],
                                     exclude_keys=["explanation", "rating"])

    df_mod = pd.DataFrame(
        [list(mod_tags.values())],
        columns=pd.MultiIndex.from_product(
            [["Moderator Results"], list(mod_tags.keys())]
        ),
    )
    df_classifier = pd.DataFrame(
        [list(classifier_tags.values())],
        columns=pd.MultiIndex.from_product(
            [["Hate Speech Classifier"], list(classifier_tags.keys())]
        ),
    )
    df_rw = pd.DataFrame(
        [list(rw_indicator.values())],
        columns=pd.MultiIndex.from_product(
            [["Right Wing Rater"], list(rw_indicator.keys())]))

    df_results = (
        pd.concat([df_classifier, df_rw, df_mod], axis=1)
        .rename(columns={0: "Evaluation"})
        .sort_index()
    )
    df_expl = (
        pd.DataFrame.from_dict(explanations)
        .transpose()
        .rename(columns={0: "Classification", 1: "Explanation"})
    )
    return df_results, df_expl


def pretty_tables_single_result(result: dict):
    df_results, df_expl = create_tables_single_result(result)
    results_styler = (
        df_results.style.map(lambda x: f"color: {'red' if x else 'grey'}")
        .hide()
        .format(lambda x: _bool_to_dot(x))
        .set_table_styles(RESULTS_STYLE)
    )

    return results_styler, df_expl
