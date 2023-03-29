"""
Make the .slurm and .sh scripts used for longleaf.
To do uncertainty analyses
"""

import argparse
from glob import glob
import os


def main(args):
    print(f"creating slurm scripts for timestamp: {args.timestamp}")
    results_dir = f'../../uncertainty_analysis_data/uncertainty_analysis_{args.timestamp}'
    longban_param_dir = os.path.join(results_dir, 'long_term_menthol_ban_parameter_sets')
    longban_options_dirs = sorted(glob(os.path.join(longban_param_dir, f'option_*')))
    mort_sets_dir = os.path.join(results_dir, 'mortality_parameter_sets')
    init_pop_dir = os.path.join(results_dir, 'initial_populations')

    num_options = len(longban_options_dirs) + 1
    num_banparams = len(glob(longban_options_dirs[0] + '/*'))
    num_mortparams = len(glob(mort_sets_dir + '/*'))
    num_initpops = len(glob(init_pop_dir + '/*'))

    filenames = []
    for opt in range(num_options):
        for half in (0,1):
            filename = f"UA_opt_{opt}_half_{half}.slurm"
            filenames.append(filename)

            file_content = """#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH --mem 10240
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 11-
#SBATCH --job
"""
            if args.email:
                file_content += f"""#SBATCH --mail-type=end
#SBATCH --mail-user={args.email}
"""
            if args.user:
                file_content += f"""#SBATCH --output=/pine/scr/{args.user[0]}/{args.user[1]}/{args.user}/UA-output-{args.timestamp}-job-%j.out

"""
            else:
                raise Exception("no longleaf user given")
            
            file_content += f"""cd /pine/scr/{args.user[0]}/{args.user[1]}/{args.user}/Gillings_work/menthol-model/mentholmodel
source activate mentholmodel_env
python uncertainty_analysis_do_runs.py {num_mortparams} {num_initpops} {num_banparams} {opt} {args.timestamp}"""

            if half:
                file_content += " --second_half"

            print(file_content)

            if os.path.isfile(f"{filename}"):
                os.remove(f"{filename}")

            with open(f"{filename}", 'w') as f:
                f.write(file_content)

            os.system(f"mv {filename} ~")

    if os.path.isfile(f"run_all_UA_{args.timestamp}.sh"):
        os.remove(f"run_all_UA_{args.timestamp}.sh")

    # os.system(f"touch ~/run_all_UA_{args.timestamp}.sh")
    with open(f"run_all_UA_{args.timestamp}.sh", "w") as f:
        for name in filenames:
            f.write("sbatch " + name + "\n")

    os.system(f"mv run_all_UA_{args.timestamp}.sh ~")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify simulation parameters')
    parser.add_argument('timestamp', 
                        type=str,
                        default='',
                        help='timestamp of uncertainty analysis directory we are working in')
    parser.add_argument('user', 
                        type=str,
                        default='',
                        help='longleaf ONYEN username (required)')
    parser.add_argument('email', 
                        type=str,
                        default='',
                        help='email for updates')
    main(parser.parse_args())