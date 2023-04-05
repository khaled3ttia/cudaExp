import random
import argparse
def generate_random_numbers(n, start, end, output_file):
    with open(output_file, 'w') as f:
        for i in range(n):
            random_number = random.uniform(start, end)
            f.write(str(random_number)+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=False, default=100000, help='Total number of numbers to generate.')
    parser.add_argument('--s', type=int, required=False, default=0, help='Range start for the generated numbers.')
    parser.add_argument('--e', type=int, required=False, default=10000, help='Range end for the generated numbers.')
    parser.add_argument('--outfile', type=str, required=False, default='input.txt',help='Output file path.')


    args = parser.parse_args()

    generate_random_numbers(args.n, args.s, args.e, args.outfile)
