"""Metadata-based recording holdouts; never inspect signals, features or scores."""

from dataclasses import asdict, dataclass, replace
from itertools import combinations
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from .results import ValidationFold


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError('Install recursive-selection[recording] for metadata splits.') from exc
    return pd


def _scalar(value):
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
        raise ValueError('Sample and condition IDs must be nonempty strings or integers.')
    return value


def _integers(values, name):
    pd = _pandas()
    values = np.asarray(pd.to_numeric(values, errors='raise'))
    if (values.dtype.kind not in 'iu' and
            (values.dtype.kind != 'f' or not np.isfinite(values).all()
             or not np.equal(values, np.floor(values)).all())):
        raise ValueError(f'{name} must contain finite integer IDs.')
    return values.astype(np.int64)


def _validation_pairs(eligible, rng, distance):
    pairs = [pair for pair in combinations(sorted(eligible), 2) if pair[1]-pair[0] >= distance]
    schedules = [schedule for schedule in combinations(pairs, 4)
        if sorted(sum(group in pair for pair in schedule) for group in eligible) == [1,1,1,1,2,2]]
    if not schedules:
        raise ValueError('No four balanced distinct Validation pairs satisfy the group distance.')
    selected = int(rng.integers(len(schedules)))
    return tuple(schedules[selected][int(i)] for i in rng.permutation(4))


@dataclass(frozen=True)
class RecordingSample:
    sample_id: str | int
    condition_id: str | int
    recording_id: str
    segment_id: int
    setup_number: int
    source_datetime: str | None = None


@dataclass(frozen=True)
class SetupRecordingSplit:
    """Nine ordered groups, fixed interior Test, four balanced repeated holdouts.

    Setup mode consumes existing zero/one-based group IDs. Time mode creates
    nine near-equal chronological recording groups independently per condition.
    Column names are configurable; metadata rows must align with X and y.
    """

    mode: str = 'setup'
    seed: int = 42
    cv_seed: int = 42
    condition_column: str = 'condition_id'
    sample_column: str = 'sample_id'
    recording_column: str = 'recording_id'
    segment_column: str = 'segment_id'
    setup_column: str = 'setup_id'
    setup_index_base: int = 0
    time_column: str = 'source_datetime'
    segments_per_recording: int | None = 10
    recording_duration_seconds: int = 600
    test_groups: tuple[int, ...] = (3,4,5,6,7)
    validation_groups: tuple[int, ...] = (2,3,4,5,6,7,8)
    min_validation_distance: int = 2

    def __post_init__(self):
        if self.mode not in ('setup', 'time', 'predefined'):
            raise ValueError('mode must be setup, time or predefined.')
        for name in ('seed', 'cv_seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError('Seeds must be nonnegative integers.')
            object.__setattr__(self, name, int(value))
        if (not isinstance(self.setup_index_base, Integral)
                or self.setup_index_base not in (0, 1) or isinstance(self.setup_index_base, bool)):
            raise ValueError('setup_index_base must be 0 or 1.')
        object.__setattr__(self, 'setup_index_base', int(self.setup_index_base))
        for name in ('min_validation_distance', 'recording_duration_seconds', 'segments_per_recording'):
            value = getattr(self, name)
            if value is None and name == 'segments_per_recording':
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
            object.__setattr__(self, name, int(value))
        for name in ('test_groups', 'validation_groups'):
            groups = tuple(getattr(self, name))
            if (not groups or len(set(groups)) != len(groups)
                    or any(isinstance(g, bool) or not isinstance(g, Integral) or not 1 <= g <= 9 for g in groups)):
                raise ValueError(f'{name} must be distinct group numbers in 1..9.')
            object.__setattr__(self, name, tuple(sorted(map(int, groups))))
        if any(len(set(self.validation_groups)-{test}) != 6 for test in self.test_groups):
            raise ValueError('Every eligible Test must leave six eligible Validation groups.')

    @classmethod
    def z24(cls, *, seed=42, cv_seed=None, **options):
        return cls(seed=seed, cv_seed=seed if cv_seed is None else cv_seed, **options)

    @classmethod
    def openbridgelab(cls, *, seed=42, cv_seed=None, **options):
        config = dict(mode='time', condition_column='label', sample_column='global_sample_id',
                      recording_column='recording_key')
        config.update(options)
        return cls(seed=seed, cv_seed=seed if cv_seed is None else cv_seed, **config)

    def _normalize(self, metadata):
        pd = _pandas()
        frame = pd.DataFrame(metadata).copy()
        columns = [self.sample_column, self.condition_column, self.recording_column, self.segment_column]
        if self.mode == 'setup': columns.append(self.setup_column)
        if self.mode == 'time': columns.append(self.time_column)
        if not set(columns).issubset(frame) or frame[columns].isna().any().any():
            raise ValueError('Missing recording identities, group IDs or timestamps in metadata.')
        if 'verified' in frame and not frame.verified.astype(str).str.lower().eq('true').all():
            raise ValueError('Unverified recording identities.')
        ids = tuple(_scalar(v) for v in frame[self.sample_column])
        if len(set(ids)) != len(ids) or not ids:
            raise ValueError('Metadata needs nonempty unique sample IDs.')
        conditions = tuple(_scalar(v) for v in frame[self.condition_column])
        if len({type(c) for c in conditions}) != 1:
            raise ValueError('Condition IDs must have a consistent sortable type.')
        recordings = tuple(str(v) for v in frame[self.recording_column])
        if any(not v.strip() for v in recordings): raise ValueError('Empty recording ID.')
        segments = _integers(frame[self.segment_column], 'segment_id')
        has_groups = self.mode == 'setup' or (self.mode == 'predefined' and self.setup_column in frame)
        setups = (_integers(frame[self.setup_column], 'setup_id')+1-self.setup_index_base
                  if has_groups else np.zeros(len(frame), dtype=int))
        times = (pd.to_datetime(frame[self.time_column], errors='raise') if self.mode == 'time' else None)
        if times is not None and times.isna().any(): raise ValueError('Missing source timestamps.')
        rows = tuple(RecordingSample(sample, condition, record, int(segment), int(setup),
            time.isoformat() if times is not None else None)
            for sample, condition, record, segment, setup, time in zip(
                ids, conditions, recordings, segments, setups, times if times is not None else [None]*len(frame)))
        seen = set()
        for row in rows:
            key = (row.condition_id, row.recording_id, row.segment_id)
            if key in seen: raise ValueError('Duplicate recording segment.')
            seen.add(key)
        records = {}
        for row in rows: records.setdefault((row.condition_id, row.recording_id), []).append(row)
        for members in records.values():
            if len({(r.setup_number, r.source_datetime) for r in members}) != 1:
                raise ValueError('A recording spans multiple groups or timestamps.')
            if self.segments_per_recording is not None:
                if sorted(r.segment_id for r in members) != list(range(self.segments_per_recording)):
                    raise ValueError('Incomplete recording segments.')
        if self.mode == 'time':
            group_ids = {}
            for condition in sorted(set(conditions)):
                keys = [k for k in records if k[0] == condition]
                keys.sort(key=lambda k: (pd.Timestamp(records[k][0].source_datetime), k[1]))
                stamps = pd.to_datetime([records[k][0].source_datetime for k in keys])
                if len(keys) < 9 or len(set(stamps)) != len(keys):
                    raise ValueError('Time groups need at least nine unambiguously ordered recordings per condition.')
                if len(stamps) > 1 and np.any((stamps[1:]-stamps[:-1]).total_seconds() < self.recording_duration_seconds):
                    raise ValueError('Overlapping nominal recording intervals.')
                for group, positions in enumerate(np.array_split(np.arange(len(keys)), 9), 1):
                    for position in positions: group_ids[keys[int(position)]] = group
            rows = tuple(replace(r, setup_number=group_ids[(r.condition_id, r.recording_id)]) for r in rows)
        return rows

    def build(self, metadata, y, *, parent_metadata=None, label_mapping=None):
        if self.mode == 'predefined':
            raise ValueError('Use RecordingSplit.from_indices() for predefined membership.')
        active = self._normalize(metadata)
        parent = self._normalize(parent_metadata) if parent_metadata is not None else active
        parent_by_id = {r.sample_id: r for r in parent}
        if not set(r.sample_id for r in active).issubset(parent_by_id):
            raise ValueError('Active sample IDs are absent from parent metadata.')
        for row in active:
            source = parent_by_id[row.sample_id]
            if (row.condition_id, row.recording_id, row.segment_id, row.source_datetime) != (
                    source.condition_id, source.recording_id, source.segment_id, source.source_datetime):
                raise ValueError('Active metadata differs from the parent identities.')
        # A subset must contain whole source conditions. Allocate ALL parent
        # conditions first; filtering must not restart either random generator.
        active_ids = {r.sample_id for r in active}
        active_conditions = {r.condition_id for r in active}
        if active_ids != {r.sample_id for r in parent if r.condition_id in active_conditions}:
            raise ValueError('An active condition must retain all its parent recordings/segments.')
        mapping = {_scalar(k):_scalar(v) for k,v in dict(label_mapping or {}).items()}
        targets = tuple(_scalar(v) for v in np.asarray(y).reshape(-1))
        expected = tuple(mapping.get(r.condition_id, r.condition_id) for r in active)
        if targets != expected or len(set(targets)) < 2:
            raise ValueError('y must align with metadata and its explicit source-to-model label map.')
        if len(set(expected)) != len(active_conditions): raise ValueError('Label mapping merges conditions.')
        outer_rng, cv_rng = np.random.default_rng(self.seed), np.random.default_rng(self.cv_seed)
        allocation = {}
        for condition in sorted({r.condition_id for r in parent}):
            rows = [r for r in parent if r.condition_id == condition]
            if {r.setup_number for r in rows} != set(range(1,10)):
                raise ValueError('Each condition needs exactly nine nonempty ordered groups.')
            test = int(outer_rng.choice(self.test_groups))
            eligible = tuple(sorted(set(self.validation_groups)-{test}))
            allocation[condition] = (test, _validation_pairs(eligible, cv_rng, self.min_validation_distance))
        active = tuple(parent_by_id[r.sample_id] for r in active)
        test_rows = tuple(i for i,r in enumerate(active) if r.setup_number == allocation[r.condition_id][0])
        test_set = set(test_rows)
        development = tuple(i for i in range(len(active)) if i not in test_set)
        folds = []
        for split in range(4):
            validation = tuple(j for j,i in enumerate(development)
                               if active[i].setup_number in allocation[active[i].condition_id][1][split])
            val_set = set(validation)
            train = tuple(j for j in range(len(development)) if j not in val_set)
            folds.append(ValidationFold(train, validation))
        result = RecordingSplit(self, parent, tuple(r.sample_id for r in active), targets,
            tuple(mapping.items()), development, test_rows, tuple(folds))
        result._validate_membership()
        return result


@dataclass(frozen=True)
class RecordingSplit:
    """Immutable input-row indices, source IDs, parent metadata and development folds."""

    policy: SetupRecordingSplit
    parent_metadata: tuple[RecordingSample, ...]
    sample_ids: tuple
    model_labels: tuple
    label_mapping: tuple
    development_rows: tuple[int, ...]
    test_rows: tuple[int, ...]
    folds: tuple[ValidationFold, ...]

    @property
    def active_metadata(self):
        lookup = {r.sample_id: r for r in self.parent_metadata}
        return tuple(lookup[sample] for sample in self.sample_ids)

    @property
    def development_groups(self):
        rows = self.active_metadata
        return tuple(json.dumps([rows[i].condition_id, rows[i].recording_id]) for i in self.development_rows)

    @property
    def validation_counts(self):
        counts = np.zeros(len(self.development_rows), dtype=int)
        for fold in self.folds: counts[list(fold.validation_rows)] += 1
        return tuple(map(int, counts))

    def _validate_membership(self):
        from .fixed_cv import validate_external_folds
        n = len(self.sample_ids)
        if n != len(self.model_labels) or len(set(self.sample_ids)) != n:
            raise ValueError('Recording split sample/label identities differ.')
        parent_ids = [row.sample_id for row in self.parent_metadata]
        if len(set(parent_ids)) != len(parent_ids) or not set(self.sample_ids).issubset(parent_ids):
            raise ValueError('Recording split parent sample identities differ.')
        parts = [np.asarray(self.development_rows), np.asarray(self.test_rows)]
        if any(p.ndim != 1 or not len(p) or p.dtype.kind not in 'iu' for p in parts):
            raise ValueError('Development/Test row indices must be nonempty integers.')
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(n)):
            raise ValueError('Development/Test must partition input rows exactly once.')
        active = self.active_metadata
        groups = [json.dumps([r.condition_id,r.recording_id]) for r in active]
        if set(groups[i] for i in self.development_rows) & set(groups[i] for i in self.test_rows):
            raise ValueError('A recording overlaps development and Test.')
        targets = np.asarray(self.model_labels)
        if not np.array_equal(np.unique(targets[list(self.test_rows)]), np.unique(targets)):
            raise ValueError('Test must contain every active model class.')
        validate_external_folds(self.folds, targets[list(self.development_rows)],
            np.asarray(self.development_groups), fold_kind='repeated')

    def validate(self):
        self._validate_membership()
        if self.policy.mode != 'predefined':
            pd = _pandas()
            def frame(rows):
                p = self.policy
                return pd.DataFrame([{p.sample_column:r.sample_id, p.condition_column:r.condition_id,
                    p.recording_column:r.recording_id, p.segment_column:r.segment_id,
                    p.setup_column:r.setup_number-1+p.setup_index_base, p.time_column:r.source_datetime} for r in rows])
            replay = self.policy.build(frame(self.active_metadata), self.model_labels,
                parent_metadata=frame(self.parent_metadata), label_mapping=dict(self.label_mapping))
            if replay != self: raise ValueError('Recording split differs from its metadata/seed recipe.')
        return self

    @classmethod
    def from_indices(cls, metadata, y, *, development_rows, test_rows, cv_splits,
                     policy=None):
        """Reuse baseline input-row indices; cv_splits indexes development locally."""
        policy = replace(policy or SetupRecordingSplit(), mode='predefined')
        rows = policy._normalize(metadata)
        from .fixed_cv import validate_external_folds
        # Preserve caller integer types until validation (never silently round floats).
        for part in (development_rows, test_rows):
            values = np.asarray(part)
            if (values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values)
                    or np.any(values < 0) or np.any(values >= len(rows))):
                raise ValueError('Development/Test indices must be nonempty in-range integers.')
        folds, _ = validate_external_folds(cv_splits, np.asarray(y)[list(development_rows)], fold_kind='repeated')
        result = cls(policy, rows, tuple(r.sample_id for r in rows), tuple(_scalar(v) for v in y), (),
            tuple(map(int, development_rows)), tuple(map(int, test_rows)), folds)
        result.validate()
        return result

    def to_config(self):
        return asdict(self)

    @classmethod
    def from_config(cls, data):
        result = cls(SetupRecordingSplit(**data['policy']),
            tuple(RecordingSample(**row) for row in data['parent_metadata']),
            tuple(data['sample_ids']), tuple(data['model_labels']), tuple(tuple(pair) for pair in data['label_mapping']),
            tuple(data['development_rows']), tuple(data['test_rows']),
            tuple(ValidationFold(tuple(f['train_rows']),tuple(f['validation_rows'])) for f in data['folds']))
        return result.validate()

    def membership_table(self):
        pd = _pandas()
        dev_lookup = {row:i for i,row in enumerate(self.development_rows)}
        records = []
        for i,row in enumerate(self.active_metadata):
            dev = dev_lookup.get(i)
            roles = ['Test' if dev is None else 'Validation' if dev in f.validation_rows else 'Train' for f in self.folds]
            records.append(dict(input_row=i, **asdict(row), model_label=self.model_labels[i],
                Pool='Test' if dev is None else 'Development', validation_count=roles.count('Validation'),
                **{f'Split_{j}':role for j,role in enumerate(roles,1)}))
        return pd.DataFrame(records)

    def save(self, directory):
        """Export recipe, row/source IDs and membership; never write signal arrays."""
        self.validate()
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()): raise FileExistsError('Choose an empty split directory.')
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'split.json').write_text(json.dumps(self.to_config(), indent=2, allow_nan=False)+'\n', encoding='utf-8')
        self.membership_table().to_csv(directory/'membership.csv', index=False)
        arrays = dict(development_rows=np.asarray(self.development_rows), test_rows=np.asarray(self.test_rows))
        for i,fold in enumerate(self.folds,1):
            arrays[f'fold_{i}_train_rows'] = np.asarray(fold.train_rows)
            arrays[f'fold_{i}_validation_rows'] = np.asarray(fold.validation_rows)
        np.savez(directory/'row_indices.npz', **arrays)
        return directory

    @classmethod
    def load(cls, directory):
        return cls.from_config(json.loads((Path(directory)/'split.json').read_text(encoding='utf-8')))
