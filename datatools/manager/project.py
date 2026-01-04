import os
from typing import Optional
from datatools.utils import PathFormatter, exists_or_make



class ProjectManager:

    PROJECT_ROOT = PathFormatter.format('/data/dataset2/Workshop')
    WEIGHT_ROOT = PathFormatter.format('/data/dataset2/TrainLog/weights')
    MODEL_DIR_MAPPING = dict(cls='defect_cls',
                             defect_cls='defect_cls',
                             seg='seg_withref',
                             seg_withref='seg_withref',
                             seg_noref='seg_noref',
                             noref='seg_noref',
                             compseg='compseg',
                             comp_seg='compseg',
                             zone_cls='zone_cls')

    def __init__(self, project, user=None):
        if user is None:
            user = os.getcwd().split(os.sep)[2]
        self.user = user
        self.project = project

    def dataset_path(self, name: str, make_dir=False, review=False):
        dataset_dir = os.path.join(ProjectManager.PROJECT_ROOT, self.user, self.project, 'data_history', name)
        if review:
            PathFormatter.review_dir(dataset_dir)
        return exists_or_make(path=dataset_dir, make_dir=make_dir)

    def redetect_path(self, dataset_name: str, trial: str, make_dir=False, review=False):
        redetect_dir = os.path.join(ProjectManager.PROJECT_ROOT, self.user, self.project,
                                    'redetect_result', dataset_name, trial)
        if review:
            PathFormatter.review_dir(redetect_dir)
        return exists_or_make(path=redetect_dir, make_dir=make_dir)

    def model_update_path(self, update: str, model: str, make_dir=False, review=False):
        assert model in ProjectManager.MODEL_DIR_MAPPING, \
            f"Model [{model}] is not allowed! Please config \'MODEL_DIR_MAPPING\'"
        update_dir = os.path.join(ProjectManager.WEIGHT_ROOT, self.user, self.project,
                                  ProjectManager.MODEL_DIR_MAPPING[model], update)
        if review:
            PathFormatter.review_dir(update_dir)
        return exists_or_make(path=update_dir, make_dir=make_dir)

    def build(self):
        models = ['cls', 'seg', 'noref', 'compseg', 'zone_cls']
        project_folders = ['data_history', 'redetect_result', 'update_records']

        weight_root = os.path.join(ProjectManager.WEIGHT_ROOT, self.user, self.project)
        project_root = os.path.join(ProjectManager.PROJECT_ROOT, self.user, self.project)
        for model in models:
            for target_dir in [weight_root,
                               os.path.join(project_root, 'train_data')]:
                os.makedirs(os.path.join(target_dir, ProjectManager.MODEL_DIR_MAPPING[model]), exist_ok=True)

        for project_folder in project_folders:
            os.makedirs(os.path.join(project_root, project_folder), exist_ok=True)

        PathFormatter.review_dir(weight_root)
        PathFormatter.review_dir(project_root)



