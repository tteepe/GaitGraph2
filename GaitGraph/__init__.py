MMK_LOGO = """  |'''''''''''''╔╬╬╬╬╬╬╬╬   _____ ______       _____ ______       ___  __
  |            ╔╬╬╬╬╬╬╬╬╬  |\   _ \  _   \    |\   _ \  _   \    |\  \|\  \\
  | ░░         ╬╬╬╬╬╬╬╬╬╬  \ \  \\\\\\__\ \  \   \ \  \\\\\\__\ \  \   \ \  \/  /|_
   ░░░░        ╬╬╬╬╬╬╬╬╬╬   \ \  \\\\|__| \  \   \ \  \\\\|__| \  \   \ \   ___  \\
  ░░░░░╦╬╦    ╔╬╬╬╬╬╬╬╬╬╬    \ \  \    \ \  \   \ \  \    \ \  \   \ \  \\\\ \  \\
 ░░░░░╬╬╬╬ ▓▓└╬╬╬╬╬╬╬╬╬╬╬     \ \__\    \ \__\   \ \__\    \ \__\   \ \__\\\\ \__\\
░░░░░╔╬╬╬ ▓▓▓  ╓╬╬╬╬╬╬╬╬╬      \|__|     \|__|    \|__|     \|__|    \|__| \|__|
░░░░░╠╬╬╬ ▓▓▓  └╬╬╬╬╬╬╬╬╬
 ░░░░└╬╬╬╬ ▓▓   ╬╬╬╬╬╬╬╬╬  Chair of Human-Machine Communication
 ░░░░░╙╬╬╬╩            ╬╬  TUM School of Computation, Information and Technology
  ░░░░░ ╚ '''''''''''''''  Technical University of Munich
   ░░░
"""


def nice_print(msg, last=False):
    print()
    print("\033[0;34m" + msg + "\033[0m")
    if last:
        print()


def cli_logo():
    nice_print(MMK_LOGO)