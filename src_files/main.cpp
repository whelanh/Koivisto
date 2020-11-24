
/****************************************************************************************************
 *                                                                                                  *
 *                                     Koivisto UCI Chess engine                                    *
 *                           by. Kim Kahre, Finn Eggers and Eugenio Bruno                           *
 *                                                                                                  *
 *                 Koivisto is free software: you can redistribute it and/or modify                 *
 *               it under the terms of the GNU General Public License as published by               *
 *                 the Free Software Foundation, either version 3 of the License, or                *
 *                                (at your option) any later version.                               *
 *                    Koivisto is distributed in the hope that it will be useful,                   *
 *                  but WITHOUT ANY WARRANTY; without even the implied warranty of                  *
 *                   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
 *                           GNU General Public License for more details.                           *
 *                 You should have received a copy of the GNU General Public License                *
 *                 along with Koivisto.  If not, see <http://www.gnu.org/licenses/>.                *
 *                                                                                                  *
 ****************************************************************************************************/

#include "Bitboard.h"
#include "Board.h"
#include "Move.h"
#include "MoveOrderer.h"
#include "Tuning.h"
#include "Verification.h"
#include "fecppnn/Data.h"
#include "fecppnn/Network.h"
#include "fecppnn/config.h"
#include "fecppnn/optimiser.h"
#include "fecppnn/sample.h"
#include "fecppnn/structure.h"
#include "uci.h"

#include <iomanip>
#include <omp.h>

using namespace std;
using namespace bb;
using namespace move;

// TODO this should be in the nn namespace, probably
//      with a better name, I think
struct training_example {
    nn::Sample input;
    nn::Data target;
};

std::vector<training_example> load_samples_from_file() {
    std::vector<training_example> v;

    tuning::loadPositionFile("../resources/other/E12.33-1M-D12-Resolved.book", 10000000);
//    tuning::loadPositionFile("../resources/other/E12.41-1M-D12-Resolved.book", 10000000);
//    tuning::loadPositionFile("../resources/other/E12.46FRC-1250k-D12-1s-Resolved.book", 10000000);
    
    int counter = 0;
    for (tuning::TrainingEntry& en : tuning::training_entries) {
        nn::Sample inSample {};
        nn::Data   target {1, 0};

        target(0) = en.target;
        for(Piece p = WHITE_PAWN; p <= BLACK_KING; p++){
            U64 bb = en.board.getPieces()[p];
            while(bb){
                Square s = bitscanForward(bb);
                inSample.indices.push_back(p * 64 + s);
                bb = lsbReset(bb);
            }
        }
        if(counter % 10000 == 0){
            std::cout << "\rconverted" << setw( 4) << counter << std::flush;;
        }
        counter ++;
    
        training_example e = {
            inSample,
            target
        };
    
        v.push_back(e);
        
    }
    std::cout << std::endl;

    return v;
}

void convertFens(const std::string& infile, const std::string& outfile){
    fstream data;
    data.open(infile, ios::in);
    if (data.is_open()) {
        ofstream outf;
        outf.open(outfile);
    
        string tp;
        int counter = 0;
        startMeasure();
        while (getline(data, tp)) {
            std::vector<std::string> splits{};
            splitString(tp, splits, ';');
            if(splits[1].at(0) == '#'){
                continue;
            }
            float res = std::stof(splits[1]);
            if(abs(res) > 10){
                continue;
            }
            Board b{splits[0]};
            
            for(Piece p = WHITE_PAWN; p <= BLACK_KING; p++){
                U64 bb = b.getPieces()[p];
                while(bb){
                    Square s = bitscanForward(bb);
                    outf << (p * 64 + s) << " ";
                    bb = lsbReset(bb);
                }
            }
            counter ++;
            if(counter % 100000 == 0){
                int time = stopMeasure();
                std::cout << "\rpositions:" << std::right << std::setw(10) << counter
                          << "   pps:" <<  std::right << std::setw(10) << 100.0 / time << "Mpps" << std::flush;
                startMeasure();
            }
            outf << res << std::endl;
        }
        outf.close();
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    
//    bb_init();
//    convertFens(argv[1], argv[2]);
//
    if (argc == 1) {
        uci_loop(false);
    } else if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        uci_loop(true);
    }
//    nn::Network net{};
//    nn::Sample sample{};
//    sample.indices.push_back(8);sample.indices.push_back(9);sample.indices.push_back(10);sample.indices.push_back(11);sample.indices.push_back(13);sample.indices.push_back(14);sample.indices.push_back(15);sample.indices.push_back(28);sample.indices.push_back(65);sample.indices.push_back(70);sample.indices.push_back(130);sample.indices.push_back(133);sample.indices.push_back(192);sample.indices.push_back(199);sample.indices.push_back(259);sample.indices.push_back(324);sample.indices.push_back(432);sample.indices.push_back(433);sample.indices.push_back(434);sample.indices.push_back(435);sample.indices.push_back(436);sample.indices.push_back(437);sample.indices.push_back(438);sample.indices.push_back(439);sample.indices.push_back(505);sample.indices.push_back(510);sample.indices.push_back(570);sample.indices.push_back(573);sample.indices.push_back(632);sample.indices.push_back(639);sample.indices.push_back(699);sample.indices.push_back(764);
//    net.compute(&sample,0);
//    std::cout << net.getOutput(0) << std::endl;

    
    return 0;
}
