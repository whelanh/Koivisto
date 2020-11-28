from glob import glob
import os

try:
    nets = glob("nn.*.bin")

    main_f = open("src_files/main.cpp")
    old_main      = main_f.read()
    main_f.close()

    uci_f = open("src_files/uci.cpp")
    old_uci = uci_f.read()
    uci_f.close()

    structure_f = open("src_files/fecppnn/structure.h")
    old_structure      = structure_f.read()
    structure_f.close()

    for net in nets:
        print(f"Compiling {net}...")

        main_f = open("src_files/main.cpp", "w")
        open("src_files/main.cpp", "w").write(old_main)
        main_f.close()

        uci_f = open("src_files/uci.cpp", "w")
        uci_f.write(old_uci)
        uci_f.close()

        structure_f = open("src_files/fecppnn/structure.h", "w")
        structure_f.write(old_structure)
        structure_f.close()

        os.environ["MAKEFLAGS"] = "-j 16"
        os.system("cmake .")
        os.system("make")
        os.system(f'mv Koivisto.exe Koivisto.{net.replace(".bin", "")}.exe')
finally:
    print(f"Reverting files...")
    main_f = open("src_files/main.cpp", "w")
    open("src_files/main.cpp", "w").write(old_main.replace("TEST = false", "TEST = true"))
    main_f.close()

    uci_f = open("src_files/uci.cpp", "w")
    uci_f.write(old_uci.replace("Koivisto 64", f"Koivisto 64 ({net})"))
    uci_f.close()

    structure_f = open("src_files/fecppnn/structure.h", "w")
    structure_f.write(old_structure.replace("nn.placeholder.bin", net))
    structure_f.close()


TEMPLATE_BASE = r"""cutechess-cli.exe -tournament gauntlet -pgnout tournament.pgn -resign movecount=5 score=1000 -concurrency 8 -recover -repeat -games 2 -rounds 4096 -openings file=C:\Users\io\Downloads\8moves_v3.pgn order=random ^
"""
TEMPLATE_ENGINE = r"""    -engine name=ENGINE_NAME cmd=ENGINE_NAME proto=uci option.Threads=1 tc=10+0.1 ^
"""

tournament_str = TEMPLATE_BASE
print("Making tournament script...")
for net in nets:
    print(f"{net}...")
    tournament_str += TEMPLATE_ENGINE.replace("ENGINE_NAME", f'Koivisto.{net.replace(".bin", "")}.exe')

print("Done")
open("tournament.bat", "w").write(tournament_str)