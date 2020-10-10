#!/bin/sh

python ensemble_models.py --versions 1 10 19 28 37 46 55 64 --use_cached_outputs &&
python ensemble_models.py --versions 2 11 20 29 38 47 56 65 --use_cached_outputs &&
python ensemble_models.py --versions 3 12 21 30 39 48 57 66 --use_cached_outputs &&
python ensemble_models.py --versions 4 13 22 31 40 49 58 67 --use_cached_outputs &&
python ensemble_models.py --versions 5 14 23 32 41 50 59 68 --use_cached_outputs &&
python ensemble_models.py --versions 6 15 24 33 42 51 60 69 --use_cached_outputs &&
python ensemble_models.py --versions 7 16 25 34 43 52 61 70 --use_cached_outputs &&
python ensemble_models.py --versions 8 17 26 35 44 53 62 71 --use_cached_outputs &&
python ensemble_models.py --versions 9 18 27 36 45 54 63 72 --use_cached_outputs &&
python ensemble_models.py --versions 1 2 3 4 5 6 7 8 9 --use_cached_outputs &&
python ensemble_models.py --versions 10 11 12 13 14 15 16 17 18 --use_cached_outputs &&
python ensemble_models.py --versions 19 20 21 22 23 24 25 26 27 --use_cached_outputs &&
python ensemble_models.py --versions 28 29 30 31 32 33 34 35 36 --use_cached_outputs &&
python ensemble_models.py --versions 37 38 39 40 41 42 43 44 45 --use_cached_outputs &&
python ensemble_models.py --versions 46 47 48 49 50 51 52 53 54 --use_cached_outputs &&
python ensemble_models.py --versions 55 56 57 58 59 60 61 62 63 --use_cached_outputs
