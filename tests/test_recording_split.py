"""Metadata-only allocation, repeated holdouts and Test isolation."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from recursive_selection import (SetupRecordingSplit, RecordingSplit, RecursiveFeatureSelector,
    CVF1ReferenceQualityPolicy, CatBoostAdapter)
from recursive_selection.fixed_cv import validate_external_folds
from recursive_selection.evaluation import audit_selection
from recursive_selection.reporting import SelectionReport
from test_selection import ScriptedAdapter, TieRanker


def setup_metadata(conditions=3):
    return pd.DataFrame([dict(sample_id=c*90+s*10+i, condition_id=c, setup_id=s,
        recording_id=f'{c}:{s}', segment_id=i, verified=True)
        for c in range(conditions) for s in range(9) for i in range(10)])


def time_metadata():
    rows=[]
    for condition,n in enumerate((56,103,125)):
        for recording in range(n):
            for segment in range(10):
                rows.append(dict(global_sample_id=len(rows),label=condition,
                    recording_key=f'{condition}:{recording}',segment_id=segment,
                    source_datetime=(pd.Timestamp('2025-01-01')+pd.Timedelta(days=condition*20,minutes=recording*10)).isoformat()))
    return pd.DataFrame(rows)


# Frozen first-three-condition allocation from the Z24/OpenBridgeLAB baselines.
TESTS=(3,6,6)
PAIRS=(((2,4),(4,6),(5,8),(2,7)), ((5,8),(2,4),(3,8),(5,7)),
       ((3,7),(5,8),(2,4),(2,8)))


def test_exact_z24_group_membership_and_balanced_repetition(tmp_path):
    m=setup_metadata();s=SetupRecordingSplit.z24().build(m,m.condition_id)
    s.validate();table=s.membership_table()
    for label in range(3):
        part=table.loc[table.condition_id==label]
        assert part.loc[part.Pool=='Test','setup_number'].unique().tolist()==[TESTS[label]]
        for fold in range(1,5):
            actual=tuple(sorted(part.loc[part[f'Split_{fold}']=='Validation','setup_number'].unique()))
            assert actual==PAIRS[label][fold-1]
            assert part.loc[part.setup_number.isin([1,9]),f'Split_{fold}'].eq('Train').all()
    assert len(s.development_rows)==240 and len(s.test_rows)==30
    assert dict(zip(*np.unique(s.validation_counts,return_counts=True)))=={0:60,1:120,2:60}
    validate_external_folds(s.folds,np.asarray(s.model_labels)[list(s.development_rows)],
                            s.development_groups,fold_kind='repeated')
    with pytest.raises(ValueError,match='exactly once'):
        validate_external_folds(s.folds,np.asarray(s.model_labels)[list(s.development_rows)])
    saved=s.save(tmp_path/'split');assert RecordingSplit.load(saved)==s
    config=s.to_config();config['folds'][0]['validation_rows']=[0]
    with pytest.raises(ValueError):RecordingSplit.from_config(config)
    assert (saved/'membership.csv').is_file() and (saved/'row_indices.npz').is_file()


def test_metadata_row_permutation_keeps_source_assignments():
    m=setup_metadata();p=SetupRecordingSplit.z24();original=p.build(m,m.condition_id)
    m=m.sample(frac=1,random_state=7).reset_index(drop=True)
    shuffled=p.build(m,m.condition_id)
    left=original.membership_table().set_index('sample_id').sort_index()
    right=shuffled.membership_table().set_index('sample_id').sort_index()
    pd.testing.assert_frame_equal(left.drop(columns='input_row'),right.drop(columns='input_row'))


def test_damage_subset_preserves_parent_rng_and_source_ids(tmp_path):
    m=time_metadata();active=m.loc[m.label.isin([1,2])].reset_index(drop=True)
    y=active.label.map({1:0,2:1}).to_numpy()
    p=SetupRecordingSplit.openbridgelab()
    s=p.build(active,y,parent_metadata=m,label_mapping={1:0,2:1})
    s.validate()
    assert len(s.development_rows)==2030 and len(s.test_rows)==250
    table=s.membership_table()
    assert table.loc[table.Pool=='Test'].groupby('condition_id').setup_number.first().tolist()==[6,6]
    parent=p.build(m,m.label)
    expected=parent.membership_table().query('condition_id in [1,2]').reset_index(drop=True)
    assert table.sample_id.tolist()==expected.sample_id.tolist()
    for fold in range(1,5):assert table[f'Split_{fold}'].tolist()==expected[f'Split_{fold}'].tolist()
    fresh=p.build(active,y,label_mapping={1:0,2:1})
    assert fresh.test_rows!=s.test_rows  # Starting RNG on two classes is a different protocol.
    s.save(tmp_path/'damage');assert RecordingSplit.load(tmp_path/'damage')==s
    assert set(s.model_labels)=={0,1} and len(s.parent_metadata)==2840


@pytest.mark.parametrize('bad',['duplicate','missing','mixed','labels','group','segments','unverified','partial_parent'])
def test_bad_metadata_fails_before_training(bad):
    m=setup_metadata();parent=None;y=m.condition_id.to_numpy()
    if bad=='duplicate':m.loc[1,'sample_id']=m.loc[0,'sample_id']
    elif bad=='missing':m.loc[0,'recording_id']=None
    elif bad=='mixed':m.loc[0,'setup_id']=8
    elif bad=='labels':y=y.copy();y[0]=2
    elif bad=='group':m.loc[m.setup_id==8,'setup_id']=7
    elif bad=='segments':m=m.iloc[1:];y=y[1:]
    elif bad=='unverified':m.loc[0,'verified']=False
    elif bad=='partial_parent':parent=m.copy();m=m.iloc[10:];y=y[10:]
    with pytest.raises(ValueError):SetupRecordingSplit.z24().build(m,y,parent_metadata=parent)


def test_ambiguous_overlapping_or_missing_time_order_rejected():
    m=time_metadata();p=SetupRecordingSplit.openbridgelab()
    for delta in (0,5):
        changed=m.copy()
        changed.loc[changed.recording_key=='1:1','source_datetime']=(pd.Timestamp('2025-01-21')+pd.Timedelta(minutes=delta)).isoformat()
        with pytest.raises(ValueError):p.build(changed,changed.label)


class IsolationAdapter(ScriptedAdapter):
    def __init__(self):super().__init__();self.observed=[]
    def fit(self,model,X,y,eval_set):
        model.classes_=np.unique(y);model.n_features_in_=X.shape[1];model.rounds=5
        self.observed.extend([set(X[:,0]),set(eval_set[0][:,0])])
    def predict(self,model,X):return X[:,1].astype(int)
    def best_rounds(self,model):return model.rounds
    def refit(self,X,y,rounds):
        model=self.new_model();model.classes_=np.unique(y);model.n_features_in_=X.shape[1]
        self.observed.append(set(X[:,0]));return model


class IsolationRanker(TieRanker):
    def rank(self,adapter,model,X,y,groups):
        adapter.observed.append(set(X[:,0]));return super().rank(adapter,model,X,y,groups)


def test_auto_selection_excludes_test_from_every_fit_and_shap(tmp_path):
    m=setup_metadata(2);y=m.condition_id.to_numpy()
    X=np.column_stack([np.arange(len(m)),y,np.ones(len(m))]);original=X.copy()
    adapter=IsolationAdapter()
    selector=RecursiveFeatureSelector(model_adapter=adapter,ranker=IsolationRanker(),
        quality_policy=CVF1ReferenceQualityPolicy(),min_features=2)
    selector.fit_recording_cv(X,y,metadata=m,split_policy=SetupRecordingSplit.z24())
    test=set(selector.test_indices_)
    assert all(not test & rows for rows in adapter.observed)
    assert selector.selected_columns_==(0,1) and selector.best_f1_==1
    assert selector.cv_evaluation_.fold_kind=='repeated'
    assert selector.split_.validation_counts.count(0)==40
    audit_selection(selector.result_,feature_groups=selector.feature_groups_.as_dict(),quality_policy=selector.quality_policy,min_groups=2)
    report=SelectionReport(selector.result_,feature_groups=selector.feature_groups_.as_dict())
    report.save(tmp_path/'reports')
    assert (tmp_path/'reports/selection_recording_membership.csv').is_file()
    np.testing.assert_array_equal(X,original)
    # Supplying an already saved split reuses membership without resampling.
    selector.fit_recording_cv(X,y,split=selector.split_)
    assert selector.selected_columns_==(0,1)
    forged=replace(selector.result_,recording_split=replace(selector.split_,test_rows=(0,)))
    with pytest.raises(ValueError):audit_selection(forged,feature_groups=selector.feature_groups_.as_dict(),quality_policy=selector.quality_policy)


@pytest.mark.integration
def test_recording_selector_persistence_and_predefined_indices(tmp_path):
    pytest.importorskip('catboost')
    m=setup_metadata(2);y=m.condition_id.to_numpy();X=np.random.default_rng(7).normal(size=(len(m),3))
    selector=RecursiveFeatureSelector(model_adapter=CatBoostAdapter(params=dict(iterations=7,depth=2,
        random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)),
        quality_policy=CVF1ReferenceQualityPolicy(),min_features=3)
    selector.fit_recording_cv(X,y,metadata=m,split_policy=SetupRecordingSplit.z24())
    s=selector.split_
    existing=RecordingSplit.from_indices(m,y,development_rows=s.development_rows,test_rows=s.test_rows,cv_splits=s.folds)
    assert existing.development_rows==s.development_rows and existing.folds==s.folds
    existing.save(tmp_path/'predefined');assert RecordingSplit.load(tmp_path/'predefined')==existing
    saved=selector.save(tmp_path/'model')
    payload=json.loads((saved/'selection.json').read_text())
    assert payload['format_version']==9 and payload['backend']['early_stopping_metric']=='accuracy'
    loaded=RecursiveFeatureSelector.load(saved)
    assert loaded.split_==s and loaded.cv_evaluation_==selector.cv_evaluation_
    np.testing.assert_array_equal(loaded.predict(X[list(s.test_rows)]),selector.predict(X[list(s.test_rows)]))
    payload['recording_split']['test_rows']=[0]
    (saved/'selection.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):RecursiveFeatureSelector.load(saved)
