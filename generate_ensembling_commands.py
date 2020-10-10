# 1 10 19 28 37 46 55 64; 2 11 20 etc.
# 1 2 3 4 5 6 7 8 9; 10 11 12 ... 18; 19 ... 27; etc.

all_commands = []
for i in range(1, 10):
    versions_to_ensemble = ' '.join([str(i + (step * 9)) for step in range(8)])
    command = f'python ensemble_models.py --versions {versions_to_ensemble} --use_cached_outputs'
    all_commands.append(command)

for i in range(1, 56, 9):
    versions_to_ensemble = ' '.join([str(i + step) for step in range(9)])
    command = f'python ensemble_models.py --versions {versions_to_ensemble} --use_cached_outputs'
    all_commands.append(command)

with open('generate_ensembling_results.sh', 'w') as f:
    f.write(' &&\n'.join(all_commands))
