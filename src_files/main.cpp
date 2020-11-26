
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
#include <cmath>

using namespace std;
using namespace bb;
using namespace move;

// TODO this should be in the nn namespace, probably
//      with a better name, I think
struct training_example {
    nn::Sample input;
    nn::Data target;
};


std::vector<training_example> convertFens(const std::string& infile){
    std::vector<training_example> examples;
    examples.reserve(75 * 1000 * 1000);
    
    fstream data;
    data.open(infile, ios::in);
    if (data.is_open()) {
        string tp;
        int counter = 0;
        startMeasure();
        while (getline(data, tp)) {
            std::vector<std::string> splits{};
            
            splitString(tp, splits, ';');
            if(splits.size() < 2) {
                continue;
            }
            
            float res;
            
            if(splits[1].at(0) == '#'){
                res = 20;
                if(splits[1].at(1) == '-') {
                    res = -res;
                }
            }
            else {
                res = std::stof(splits[1]);
                res = std::max(res, -20.0f);
                res = std::min(res,  20.0f);
            }
            
            Board b{splits[0]};
            
            nn::Sample sample {};
            nn::Data   target {1, 0};
            
            if(b.getActivePlayer() == WHITE){
                for(Piece p = WHITE_PAWN; p <= BLACK_KING; p++){
                    U64 bb = b.getPieces()[p];
                    while(bb) {
                        Square s = bitscanForward(bb);
                        sample.indices.push_back(p * 64 + s);
                        bb = lsbReset(bb);
                    }
                }
            }else{
                res = -res;
                for(Piece p = WHITE_PAWN; p <= BLACK_KING; p++){
                    U64 bb = b.getPieces()[p];
                    while(bb) {
                        Square s = mirrorSquare(bitscanForward(bb));
                        Piece p_correct = p;
                        if(p > WHITE_KING){
                            p_correct = p-6;
                        }else{
                            p_correct = p+6;
                        }
                        sample.indices.push_back(p_correct * 64 + s);
                        bb = lsbReset(bb);
                    }
                }
            }
            
    
            target(0) = res;
            training_example e = {
                sample,
                target
            };
            
            examples.push_back(e);
            counter++;
            if (counter > 1000 * 1000) {
                break;
            }
            if(counter % 100000 == 0) {
                int time = stopMeasure();
                std::cout << "\rpositions:" << std::right << std::setw(10) << counter
                          << "   pps:" <<  std::right << std::setw(10) << 100.0 / time << "Mpps" << std::flush;
                startMeasure();
            }
        }
    }
    std::cout << std::endl;
    return examples;
}

int main(int argc, char* argv[]) {

    if (argc == 1) {
        uci_loop(false);
    } else if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        uci_loop(true);
    } else if (argc > 1 && strcmp(argv[1], "train") == 0) {
        #ifdef NN_TRAIN
        bb_init();
    
        nn::Network net {};
        
        int threadID = 0;
        
        std::vector<training_example> train_dataset = convertFens("E:/test.fen");
        net.compute(&train_dataset[0].input, threadID);
        
        float lossSum = 0;
        // for each epoch, divide the data into batches and train the batches
        for(int epoch = 0; epoch < 2000; epoch++) {
            startMeasure();
            
            // batchStart is the starting index for the current batch
            for(int batchStart = 0; batchStart < train_dataset.size(); batchStart += NN_BATCH_SIZE){
                
                lossSum = 0;
    #pragma omp parallel for schedule(static, NN_BATCH_SIZE / NN_THREADS) num_threads(NN_THREADS) reduction(+:lossSum)
                for(int index = batchStart; index < batchStart + NN_BATCH_SIZE; index++){
                    if (index >= train_dataset.size()) continue;
                    const int threadID = omp_get_thread_num();
                    net.compute(&train_dataset[index].input, threadID);
                    float loss = net.applyLoss(&train_dataset[index].target, threadID);
                    net.backprop(&train_dataset[index].input, threadID);
                    lossSum += loss;
                }
                
                net.mergeGrad();
                net.optimise();
                net.clearGrad();
                
                int processedValuesEpoch = batchStart + NN_BATCH_SIZE;
                int processedValuesBatch = NN_BATCH_SIZE;
                if(batchStart + NN_BATCH_SIZE > train_dataset.size()){
                    processedValuesBatch = train_dataset.size() - batchStart;
                    processedValuesEpoch = train_dataset.size();
                }
                lossSum /= processedValuesBatch;
                
                if(batchStart % (NN_BATCH_SIZE * 10) == 0) {
                    std::cout << "\repoch" << setw(4) << epoch <<
                        "; train loss=" << setw(10) << setprecision(6) << right << lossSum <<
                        "; validation loss=" << setw(10) << setprecision(6) << right << lossSum <<
                        "; speed=" << setw(10) << setprecision(0) << right << processedValuesEpoch / stopMeasure() * 1000
                            << std::flush;
                }            
            }

            std::string fname = "nn.";
            fname += std::to_string(epoch);
            fname += ".bin";

            net.writeWeights(fname);
            net.writeWeights("nn.latest.bin");
            std::cout << std::endl;
        }
        #endif
    }

//    bb::bb_init();
//    eval_init();
//    Board b{"8/2N5/8/k2p4/8/8/2q1K3/8 w - - 2 76"};
//    Evaluator ev{};
//    std::cout << ev.evaluate(&b, true) << std::endl;
    
    
    return 0;
}
