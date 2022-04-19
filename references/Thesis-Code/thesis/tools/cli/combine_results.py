import click
import os
import json


@click.command()
@click.argument('a')
@click.argument('b')
@click.argument('output')
def combine(a, b, output):
    base = '/home/anton/git/thesis/Thesis-Code/results'
    with open(os.path.join(base, a)) as fa, open(os.path.join(base, b)) as fb, open(
            os.path.join(base, output), 'w') as fout:
        data_a = json.load(fa)
        data_b = json.load(fb)
        data_a['results'].extend(data_b['results'])
        json.dump(data_a, fout)


if __name__ == '__main__':
    combine()