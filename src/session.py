"""Dataset state transitions, independent of the Streamlit UI."""


def clear_dataset(state):
    for key in ("df", "original_df", "info", "source_id", "history", "changes"):
        state.pop(key, None)


def start_dataset(state, df, info, source_id):
    clear_dataset(state)
    state.update(df=df.copy(deep=True), original_df=df.copy(deep=True),
                 info=dict(info), source_id=source_id, history=[], changes=[])


def apply_change(state, frame, description):
    history = state["history"]
    history.append((state["df"].copy(deep=True), list(state["changes"])))
    del history[:-5]
    before = len(state["df"])
    state["df"] = frame.copy(deep=True)
    state["changes"] = state["changes"] + [f"{description}: {before} → {len(frame)} строк"]


def undo_change(state):
    if state["history"]:
        state["df"], state["changes"] = state["history"].pop()


def restore_original(state):
    state["df"] = state["original_df"].copy(deep=True)
    state["history"] = []
    state["changes"] = []
