import numpy as np
import pandas as pd
import pytest

from climate_yield.temporal_validation import TemporalSplit, assert_no_temporal_overlap, expanding_window_folds


def test_split_from_config_is_chronological(cfg):
    s = TemporalSplit.from_config(cfg["task"])
    assert s.train_years[1] < s.validation_years[0] <= s.validation_years[1] < s.test_years[0]
    years = pd.Series(np.arange(1991, 2017))
    tr, va, te = (years[s.mask(years, p)] for p in ("train", "validation", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(tr) + len(va) + len(te) == len(years)
    dev = years[s.mask(years, "development")]
    assert dev.max() < te.min()


def test_overlapping_split_rejected():
    with pytest.raises(ValueError):
        TemporalSplit((1990, 2005), (2004, 2008), (2009, 2010))
    with pytest.raises(ValueError):
        TemporalSplit((1990, 2000), (2001, 2005), (2003, 2010))


def test_expanding_window_folds_train_precedes_validation():
    years = np.repeat(np.arange(1991, 2013), 3)
    folds = expanding_window_folds(years, [[2005, 2006], [2007, 2008], [2011, 2012]])
    sizes = []
    for tr, va in folds:
        assert years[tr].max() < years[va].min()
        sizes.append(len(tr))
    assert sizes == sorted(sizes)  # expanding
    assert set(years[folds[0][1]]) == {2005, 2006}


def test_empty_fold_rejected():
    with pytest.raises(ValueError):
        expanding_window_folds(np.arange(2000, 2005), [[1990, 1991]])


def test_assert_no_overlap():
    assert_no_temporal_overlap([2000, 2001], [2002])
    with pytest.raises(AssertionError):
        assert_no_temporal_overlap([2000, 2003], [2002])
