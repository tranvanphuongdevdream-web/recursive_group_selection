"""Rising F1 reference, deterministic fold workers, and historical persistence."""
from dataclasses import replace
from types import SimpleNamespace
from threading import Barrier, Lock, get_ident
import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from recursive_selection import (CVF1ReferenceQualityPolicy, BatchRemovalOptions,
    RecursiveFeatureSelector, CatBoostAdapter, LightGBMAdapter, XGBoostAdapter,
    SklearnGBCAdapter)
from recursive_selection.results import MacroScores
from recursive_selection.cross_validation import summarize_scores
from recursive_selection.evaluation import audit_selection
from recursive_selection.reporting import SelectionReport
from recursive_selection.acceleration import SelectionSpeedOptions
from test_selection import TieRanker


def test_reference_rises_and_never_follows_a_tolerance_decrease():
    policy = CVF1ReferenceQualityPolicy()
    full = summarize_scores([MacroScores(.8,.8,.8)]*4)
    floor = policy.thresholds(full)
    assert policy.reference == 'best'
    for value, expected in [(.82, True), (.81, False), (.82, True), (.82-5e-13, True), (.83, True)]:
        score = MacroScores(.1,.2,value)
        before = floor.f1_macro
        assert policy.evaluate(score,floor).passed == expected
        if expected:floor = policy.advance(floor,score)
        assert floor.f1_macro >= before
    assert floor.f1_macro == .83
    fixed = CVF1ReferenceQualityPolicy(reference='full')
    assert fixed.advance(fixed.thresholds(full),MacroScores(1,1,1)).f1_macro == .8
    with pytest.raises(ValueError, match='reference'):
        CVF1ReferenceQualityPolicy(reference='unknown')


class ParallelAdapter:
    backend_name = 'scripted'
    def __init__(self, fail=False):
        self.params = {'nested':[1]}
        self.fit_params = {'nested':[2]}
        self.barrier, self.lock = Barrier(4), Lock()
        self.workers, self.models = [], []
        self.fail, self.refits = fail, []

    def new_model(self):
        model = SimpleNamespace()
        self.models.append(model)
        return model

    def fit(self, model, X, y, eval_set):
        fold = int(eval_set[0][0,0])//6 + 1
        self.params['nested'].append(fold)
        self.fit_params['nested'].append(fold)
        with self.lock:self.workers.append((id(self), get_ident(), id(model)))
        self.barrier.wait(timeout=15)
        if self.fail and fold == 2:raise RuntimeError('fold failed')
        model.classes_, model.n_features_in_, model.rounds = np.unique(y),X.shape[1],fold

    def refit(self, X, y, rounds):
        self.refits.append(rounds)
        return SimpleNamespace(classes_=np.unique(y),n_features_in_=X.shape[1],rounds=rounds)

    def predict(self, model, X):return np.arange(len(X)) % 2
    def classes(self, model):return model.classes_
    def feature_count(self, model):return model.n_features_in_
    def best_rounds(self, model):return model.rounds


@pytest.mark.parametrize('fail', [False, True])
def test_four_workers_isolated_and_events_stay_on_parent(monkeypatch, fail):
    monkeypatch.setattr('recursive_selection.acceleration.cpu_count',lambda:4)
    X = np.column_stack([np.arange(24),np.zeros((24,3))])
    y = np.arange(24)%2
    folds = [(np.flatnonzero(np.arange(24)//6!=i),np.arange(i*6,(i+1)*6)) for i in range(4)]
    adapter = ParallelAdapter(fail)
    parent = get_ident()
    events = []
    class Recorder:
        def record(self,event):events.append((get_ident(),event))
    selector = RecursiveFeatureSelector(model_adapter=adapter,ranker=TieRanker(),
        quality_policy=CVF1ReferenceQualityPolicy(),min_features=4,recorder=Recorder(),
        speed_options=BatchRemovalOptions(cv_jobs=4,cpu_budget=4))
    if fail:
        with pytest.raises(RuntimeError,match='fold failed'):
            selector.fit_cv(X,y,cv_splits=folds)
        assert not selector._is_fitted and not adapter.refits
    else:
        selector.fit_cv(X,y,cv_splits=folds)
        assert selector.cv_evaluation_.best_rounds[0] == (1,2,3,4)
        assert selector.refit_rounds_ == 3
        assert selector.timings_['cv_workers'] == 4
        assert [e['fold'] for _,e in events if e['event']=='cv_fold_completed'] == [1,2,3,4]
    assert len({x[0] for x in adapter.workers}) == 4
    assert len({x[1] for x in adapter.workers}) == 4
    assert len({x[2] for x in adapter.workers}) == 4
    assert adapter.params == {'nested':[1]} and adapter.fit_params == {'nested':[2]}
    assert all(thread==parent for thread,event in events)


@pytest.mark.parametrize('key', ['cv_jobs','cpu_budget'])
@pytest.mark.parametrize('value', [0,-1,True,1.5])
def test_worker_configuration_validation(key,value):
    with pytest.raises(ValueError, match=key):BatchRemovalOptions(**{key:value})


def test_cpu_budget_and_gpu_safety(monkeypatch):
    monkeypatch.setattr('recursive_selection.acceleration.cpu_count',lambda:6)
    options = BatchRemovalOptions(cv_jobs=4,cpu_budget=6)
    assert options.resources(4,SimpleNamespace(params={'n_jobs':2})) == (4,1)
    assert BatchRemovalOptions(cv_jobs=4,cpu_budget=2).resources(4,SimpleNamespace(params={})) == (2,1)
    for params in ({'task_type':'GPU'},{'device':'cuda:0'},{'device_type':'gpu'}):
        assert options.resources(4,SimpleNamespace(params=params))[0] == 1
    assert BatchRemovalOptions().cv_jobs == 1


def make_adapter(backend):
    if backend=='catboost':
        pytest.importorskip('catboost')
        return CatBoostAdapter(params=dict(iterations=10,depth=2,random_seed=42,
            loss_function='MultiClass',thread_count=1,verbose=False,allow_writing_files=False))
    if backend=='lightgbm':
        pytest.importorskip('lightgbm')
        return LightGBMAdapter(params=dict(n_estimators=10,max_depth=2,num_leaves=4,
            random_state=42,n_jobs=1,min_child_samples=2,metric='multi_logloss',verbosity=-1))
    if backend=='xgboost':
        pytest.importorskip('xgboost')
        return XGBoostAdapter(params=dict(n_estimators=10,max_depth=2,random_state=42,
            eval_metric='mlogloss',n_jobs=1,tree_method='hist'))
    pytest.importorskip('shap')
    return SklearnGBCAdapter(params=dict(n_estimators=10,max_depth=2,random_state=42))


@pytest.mark.integration
@pytest.mark.parametrize('backend',['catboost','lightgbm','xgboost','sklearn_gbc'])
def test_real_parallel_matches_sequential_and_roundtrips(backend,monkeypatch,tmp_path):
    monkeypatch.setattr('recursive_selection.acceleration.cpu_count',lambda:4)
    X,y = make_classification(n_samples=120,n_features=6,n_informative=4,
        n_redundant=0,n_classes=3,random_state=17)
    folds = list(StratifiedKFold(4,shuffle=True,random_state=42).split(X,y))
    selectors=[]
    for jobs in (1,4):
        selector = RecursiveFeatureSelector(model_adapter=make_adapter(backend),
            quality_policy=CVF1ReferenceQualityPolicy(),min_features=5,
            speed_options=BatchRemovalOptions(cv_jobs=jobs))
        spec = selector.model_adapter.specification()
        selector.fit_cv(X,y,cv_splits=folds)
        assert selector.model_adapter.specification()==spec
        audit_selection(selector.result_,feature_groups=selector.feature_groups_.as_dict(),
                        quality_policy=selector.quality_policy,min_groups=5)
        selectors.append(selector)
    seq,par=selectors
    assert seq.selected_columns_ == par.selected_columns_
    assert seq.cv_evaluation_.best_rounds == par.cv_evaluation_.best_rounds
    assert seq.cv_evaluation_.refit_rounds == par.cv_evaluation_.refit_rounds
    assert seq.cv_evaluation_.baseline.fold_scores == par.cv_evaluation_.baseline.fold_scores
    assert [(t.metrics,t.accepted,t.f1_reference_before,t.f1_reference_after) for t in seq.history_] == [
        (t.metrics,t.accepted,t.f1_reference_before,t.f1_reference_after) for t in par.history_]
    np.testing.assert_array_equal(seq.predict(X),par.predict(X))
    par.save(tmp_path/'selection')
    restored=RecursiveFeatureSelector.load(tmp_path/'selection',trusted=True)
    assert restored.quality_policy.reference=='best'
    assert restored.best_f1_==par.best_f1_
    assert restored.cv_evaluation_==par.cv_evaluation_
    report=SelectionReport(restored.result_,feature_groups=restored.feature_groups_.as_dict())
    assert 'F1 reference before' in report.trials
    report.save(tmp_path/'report')
    assert (tmp_path/'report/selection_f1_reference_history.csv').is_file()
    for bad in ('before','after','mode'):
        result=par.result_
        if not result.history:continue
        if bad=='mode':result=replace(result,cv_evaluation=replace(result.cv_evaluation,reference='full'))
        else:
            t=replace(result.history[0],**{'f1_reference_'+bad:.123})
            result=replace(result,history=(t,*result.history[1:]))
        with pytest.raises(ValueError):
            audit_selection(result,feature_groups=par.feature_groups_.as_dict(),
                            quality_policy=par.quality_policy,min_groups=5)


@pytest.mark.integration
def test_format7_loads_fixed_reference_instead_of_new_default(tmp_path):
    adapter=make_adapter('sklearn_gbc')
    X,y=make_classification(n_samples=90,n_features=6,n_informative=4,
        n_redundant=0,n_classes=3,random_state=17)
    folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(X,y))
    selector=RecursiveFeatureSelector(model_adapter=adapter,
        quality_policy=CVF1ReferenceQualityPolicy(reference='full'),min_features=5,
        speed_options=BatchRemovalOptions())
    selector.fit_cv(X,y,cv_splits=folds)
    selector.save(tmp_path/'old')
    path=tmp_path/'old/selection.json'
    data=json.loads(path.read_text())
    data['format_version']=7
    data['policy'].pop('reference')
    data['cv_evaluation'].pop('reference')
    for key in ('cv_jobs','cpu_budget'):data['speed_options'].pop(key)
    for t in data['history']:
        t.pop('f1_reference_before');t.pop('f1_reference_after')
    path.write_text(json.dumps(data))
    restored=RecursiveFeatureSelector.load(tmp_path/'old',trusted=True)
    assert restored.quality_policy.reference=='full'
    assert restored.cv_evaluation_.reference=='full'
    assert restored.speed_options.cv_jobs==1
    np.testing.assert_array_equal(restored.predict(X),selector.predict(X))
    data['policy']['reference']='best'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Legacy format 7'):
        RecursiveFeatureSelector.load(tmp_path/'old',trusted=True)
