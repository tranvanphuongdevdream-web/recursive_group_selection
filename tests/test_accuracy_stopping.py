"""Independent native ensembles verify Accuracy checkpoints, ties and cap paths."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from recursive_selection import CatBoostAdapter, LightGBMAdapter, XGBoostAdapter, SklearnGBCAdapter
from recursive_selection.backends.restoration import adapter_from_specification


@pytest.mark.integration
@pytest.mark.parametrize('backend',['catboost','lightgbm','xgboost','sklearn_gbc'])
@pytest.mark.parametrize('classes',[2,3])
@pytest.mark.parametrize('patience',[5,40])
def test_default_matches_independent_native_accuracy_checkpoint(backend,classes,patience,tmp_path):
    pytest.importorskip('sklearn' if backend=='sklearn_gbc' else backend)
    X,y=make_classification(n_samples=160,n_features=5,n_informative=3,n_redundant=0,
        n_classes=classes,flip_y=.18,random_state=17)
    X,Xv,y,yv=train_test_split(X,y,test_size=.35,random_state=3,stratify=y)
    cap=18
    if backend=='catboost':
        from catboost import CatBoostClassifier
        params=dict(iterations=cap,depth=2,learning_rate=.15,random_seed=42,
            thread_count=1,allow_writing_files=False,verbose=False)
        full=CatBoostClassifier(**params,eval_metric='Accuracy')
        full.fit(X,y,eval_set=(Xv,yv),use_best_model=False)
        scores=[accuracy_score(yv,full.predict(Xv,ntree_end=i).reshape(-1)) for i in range(1,cap+1)]
        adapter=CatBoostAdapter(params=params,early_stopping_rounds=patience)
    elif backend=='lightgbm':
        from lightgbm import LGBMClassifier
        params=dict(n_estimators=cap,max_depth=2,num_leaves=4,min_child_samples=3,
            learning_rate=.15,random_state=42,n_jobs=1,verbosity=-1)
        full=LGBMClassifier(**params).fit(X,y)
        scores=[accuracy_score(yv,full.predict(Xv,num_iteration=i)) for i in range(1,cap+1)]
        adapter=LightGBMAdapter(params=params,early_stopping_rounds=patience)
    elif backend=='xgboost':
        from xgboost import XGBClassifier
        params=dict(n_estimators=cap,max_depth=2,learning_rate=.15,random_state=42,n_jobs=1,tree_method='hist')
        full=XGBClassifier(**params).fit(X,y)
        scores=[accuracy_score(yv,full.predict(Xv,iteration_range=(0,i))) for i in range(1,cap+1)]
        adapter=XGBoostAdapter(params=params,early_stopping_rounds=patience)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        params=dict(n_estimators=cap,max_depth=2,learning_rate=.15,random_state=42)
        full=GradientBoostingClassifier(**params).fit(X,y)
        scores=[accuracy_score(yv,p) for p in full.staged_predict(Xv)]
        adapter=SklearnGBCAdapter(params=params,early_stopping_rounds=patience)
    best_score,best=-np.inf,0
    for i,score in enumerate(scores,1):
        if score>best_score:best_score,best=score,i
        if i-best>=patience:break
    original=adapter.specification()
    model=adapter.new_model();adapter.fit(model,X,y,(Xv,yv))
    assert adapter.early_stopping_metric=='accuracy'
    assert adapter.best_rounds(model)==best
    assert accuracy_score(yv,adapter.predict(model,Xv))==best_score
    assert adapter.specification()==original
    adapter.validate_full_model(model)
    if backend=='xgboost':assert model.get_booster().num_boosted_rounds()==best
    if backend=='sklearn_gbc':
        assert model.n_estimators_==best
        assert model.best_validation_accuracy_==best_score
        assert model.best_iteration_==np.argmax(model.validation_accuracy_)
    path=tmp_path/adapter.model_filename;adapter.save_model(model,path)
    loaded=adapter.load_model(path,np.unique(y))
    np.testing.assert_array_equal(adapter.predict(loaded,Xv),adapter.predict(model,Xv))


def test_legacy_specification_restores_configured_monitor():
    for backend in ('catboost','lightgbm','xgboost','sklearn_gbc'):
        adapter=adapter_from_specification(dict(name=backend,params={},fit_params={},early_stopping_rounds=5))
        assert adapter.early_stopping_metric=='configured'
        assert adapter_from_specification(adapter.specification()).specification()==adapter.specification()
    with pytest.raises(ValueError,match='early_stopping_metric'):
        CatBoostAdapter(early_stopping_metric='unknown')
