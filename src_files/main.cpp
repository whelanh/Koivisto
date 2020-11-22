
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

int main(int argc, char* argv[]) {
    bb_init();
    
    int threadID = 0;

    nn::Network net {};
    std::vector<training_example> train_dataset = load_samples_from_file();
    net.compute(&train_dataset[0].input, threadID);

    float lossSum = 0;
    // for each epoch, divide the data into batches and train the batches
    for(int epoch = 0; epoch < 2000; epoch++){
        
        // batchStart is the starting index for the current batch
        for(int batchStart = 0; batchStart < train_dataset.size(); batchStart+=NN_BATCH_SIZE){

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

            lossSum /= NN_BATCH_SIZE;

            std::cout << "\repoch" << setw(4) << epoch << "; train loss=" << setw(10) << setprecision(6) << right
                      << lossSum << "; validation loss=" << setw(10) << setprecision(6) << right << lossSum
                      << std::flush;

            lossSum = 0;
        }
        std::cout << std::endl;

//        for (int i = 0; i < train_dataset.size(); i++) {
//            net.compute(&train_dataset[i].input, threadID);
//            float loss = net.applyLoss(&train_dataset[i].target, threadID);
//            net.backprop(&train_dataset[i].input, threadID);
//            lossSum += loss;
//
//            if(i % NN_BATCH_SIZE == NN_BATCH_SIZE - 1) {
//                net.optimise();
//                net.mergeGrad();
//                net.clearGrad();
//            }
//            if(i % (100 * NN_BATCH_SIZE) == 0) {
//
//                lossSum /= 100 * NN_BATCH_SIZE;
//                std::cout << "\repoch"            << setw( 4) << epoch
//                          << "; train loss="      << setw(10) << setprecision(6) << right << lossSum
//                          << "; validation loss=" << setw(10) << setprecision(6) << right << lossSum << std::flush;
//
//                lossSum = 0;
//            }
//        }
//        std::cout << std::endl;
    }

//
//    /**********************************************************************************
//     *                                  T U N I N G                                   *
//     **********************************************************************************/
//
//    // main_tune_pst_bb(PAWN);
//    //    eval_init();
//    //     main_tune_features();
//    // main_tune_pst();
//    // main_tune_features_bb();
    
    return 0;
}
