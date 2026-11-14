"""Participant rounding, disjoint holdouts and equal-person validation scoring."""
import copy
import unittest

from scripts.subject_splits import participant_partitions
from scripts.validation_scoring import validation_selection, validation_score
from scripts.metrics import classification_metrics


class ParticipantValidationTests(unittest.TestCase):
    def test_dynamic_counts_rotate_every_test_participant(self):
        for n, expected in [(3, (1, 1, 1)), (8, (5, 2, 1)), (9, (6, 2, 1)), (10, (6, 3, 1)), (12, (8, 3, 1))]:
            people = ['person%02d' % i for i in range(n)]
            config = dict(type='leave_one_subject_out', subject_counts=dict(train=None, validation=None, test=1),
                          validation_subject_fraction=0.25)
            before = copy.deepcopy(config)
            folds = participant_partitions(people, config)
            self.assertEqual(config, before)
            self.assertEqual(folds, participant_partitions(reversed(people), config))
            self.assertEqual([fold['test'] for fold in folds], [[p] for p in people])
            for fold in folds:
                self.assertEqual(tuple(len(fold[p]) for p in ('train', 'validation', 'test')), expected)
                self.assertEqual(len(set(fold['train'] + fold['validation'] + fold['test'])), n)

    def test_count_fraction_and_fixed_id_overrides(self):
        people = list('abcdefgh')
        config = dict(type='subject_holdout', validation_subject_fraction=0.375)
        self.assertEqual(len(participant_partitions(people, config)[0]['validation']), 3)
        config['subject_counts'] = dict(train=3, validation=1, test=2)
        config['test_subject_ids'] = ['h', 'g']
        fold, = participant_partitions(people, config)
        self.assertEqual(tuple(len(fold[p]) for p in ('train','validation','test','excluded')), (3,1,2,2))
        config['subject_counts'] = dict(train=None, validation=None, test=2)
        config['validation_subject_ids'] = ['a']
        fold, = participant_partitions(people, config)
        self.assertEqual(fold['validation'], ['a'])
        self.assertEqual(len(fold['train']), 5)
        config['validation_subject_ids'] = None
        config['subject_counts']['validation'] = 0
        self.assertFalse(participant_partitions(people, config)[0]['validation'])

    def test_invalid_configuration_is_not_silently_clipped(self):
        for fraction in [0, 1, -1, True, '0.25', float('nan')]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                participant_partitions(list('abcdefgh'), dict(type='subject_holdout', validation_subject_fraction=fraction))
        with self.assertRaises(ValueError):
            participant_partitions(list('abc'), dict(type='subject_holdout', subject_counts=dict(train=2,validation=1,test=1)))

    def test_each_validation_person_has_equal_weight_for_both_tasks(self):
        actual = [0, 1] * 11
        metadata = [dict(subject_id='large')] * 20 + [dict(subject_id='small')] * 2
        first = actual[:20] + [1, 0]
        second = [0] * 20 + [0, 1]
        for labels in ({'grasp': 0, 'release': 1}, {'grasp::object=1': 0, 'grasp::object=2': 1}):
            metrics1 = classification_metrics(actual, first, labels)
            metrics2 = classification_metrics(actual, second, labels)
            selection = validation_selection({'protocol_metadata': {'type': 'leave_one_subject_out'}}, True)
            score1 = validation_score(metrics1, actual, first, metadata, labels, selection)
            score2 = validation_score(metrics2, actual, second, metadata, labels, selection)
            self.assertAlmostEqual(score1, 0.5)
            self.assertAlmostEqual(score2, 2/3)
            self.assertGreater(metrics1['macro_f1'], metrics2['macro_f1'])
            self.assertLess(score1, score2)
            self.assertEqual(validation_score(metrics1, [], [], [], labels, 'validation_macro_f1'), metrics1['macro_f1'])
            with self.assertRaises(ValueError):
                validation_score(metrics1, actual, first, [{}] * len(actual), labels, selection)
            with self.assertRaises(ValueError):
                validation_score(metrics1, actual, first, metadata[:-1], labels, selection)

    def test_selection_is_limited_to_subject_protocols(self):
        for kind in ('subject_holdout', 'leave_one_subject_out'):
            self.assertEqual(validation_selection({'protocol_metadata': {'protocol_type': kind}}, True), 'validation_subject_mean_macro_f1')
        self.assertEqual(validation_selection({'protocol_metadata': {'type': 'held_out_repetitions'}}, True), 'validation_macro_f1')
        self.assertEqual(validation_selection({}, False), 'last_epoch_no_validation')


if __name__ == '__main__':
    unittest.main()
