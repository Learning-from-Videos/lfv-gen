from dataclasses import dataclass


@dataclass
class R3MDemonstrationDataset:
    suite: str
    viewpoint: str
    task: str
    name: str
    gdrive_id: str

    @property
    def path(self) -> str:
        return f"{self.suite}/{self.viewpoint}/{self.name}"


#
R3M_DATASETS: tuple[R3MDemonstrationDataset, ...] = (
    # Metaworld datasets
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="left_cap2",
        task="button-press-topdown-v2-goal-observable",
        name="button-press-topdown-v2-goal-observable.pickle",
        gdrive_id="11XyH8D5gsm0aA-du66NLhN1F-dUuX6gt",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="left_cap2",
        task="hammer-v2-goal-observable",
        name="hammer-v2-goal-observable.pickle",
        gdrive_id="127MDie6Knn8eqysSbHjjIke0sc2h-TwC",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="left_cap2",
        task="drawer-open-v2-goal-observable",
        name="drawer-open-v2-goal-observable.pickle",
        gdrive_id="1237SZJY24IWpqEUZ4qMQOXaccciMg9Ze",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="left_cap2",
        task="bin-picking-v2-goal-observable",
        name="bin-picking-v2-goal-observable.pickle",
        gdrive_id="11W63B28cvwWaOX64YJKKpjJSZJvjrpzZ",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="left_cap2",
        task="assembly-v2-goal-observable",
        name="assembly-v2-goal-observable.pickle",
        gdrive_id="11RyG2TJp4WeifPZhoptw1g7wC8gXwwCL",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="right_cap2",
        task="bin-picking-v2-goal-observable",
        name="bin-picking-v2-goal-observable.pickle",
        gdrive_id="12CvPljKZbjJbF1DcgzThMQiVsmVaElrH",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="right_cap2",
        task="hammer-v2-goal-observable",
        name="hammer-v2-goal-observable.pickle",
        gdrive_id="1--HThjLzU7V3ZljnZXWDAzuAKtfT5S_G",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="right_cap2",
        task="drawer-open-v2-goal-observable",
        name="drawer-open-v2-goal-observable.pickle",
        gdrive_id="1UeeAIV6QCZcjgtt9G4i8brnoChrLYUfz",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="right_cap2",
        task="button-press-topdown-v2-goal-observable",
        name="button-press-topdown-v2-goal-observable.pickle",
        gdrive_id="1bYDgnj3OiQUvlXA2oO3pyEIfDpZN42X4",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="right_cap2",
        task="assembly-v2-goal-observable",
        name="assembly-v2-goal-observable.pickle",
        gdrive_id="12B6AlTWW-yyC-RQINqkcMzy8M5Jaf4xW",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="top_cap2",
        task="hammer-v2-goal-observable",
        name="hammer-v2-goal-observable.pickle",
        gdrive_id="1-3ICBIyeBjxdxjNMOPEeWhwTvc3ZSbJs",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="top_cap2",
        task="drawer-open-v2-goal-observable",
        name="drawer-open-v2-goal-observable.pickle",
        gdrive_id="1-35O2Fmh72cFi42SUWgQUHVm_QsORoI9",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="top_cap2",
        task="button-press-topdown-v2-goal-observable",
        name="button-press-topdown-v2-goal-observable.pickle",
        gdrive_id="1--ZXoUNyIU4sqLIvbEoYdpWlA8wpKHGD",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="top_cap2",
        task="bin-picking-v2-goal-observable",
        name="bin-picking-v2-goal-observable.pickle",
        gdrive_id="1-5XN1Wt5wkaKeYWGXSiVPaTKmChQxDIq",
    ),
    R3MDemonstrationDataset(
        suite="metaworld",
        viewpoint="top_cap2",
        task="assembly-v2-goal-observable",
        name="assembly-v2-goal-observable.pickle",
        gdrive_id="1-4lS1Oy6cvb8GD5nlZEZj9rnt3s-9PHA",
    ),
)
