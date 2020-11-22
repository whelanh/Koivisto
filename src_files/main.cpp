
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
#include "uci.h"

#include <iomanip>

using namespace std;
using namespace bb;
using namespace move;

// TODO kick this out of the main file...
void main_tune_features() {
    eval_init();
    bb_init();
    Evaluator* evaluator = new Evaluator();
    
    using namespace tuning;
    
    loadPositionFile("../resources/other/E12.33-1M-D12-Resolved.book", 10000000);
    loadPositionFile("../resources/other/E12.41-1M-D12-Resolved.book", 10000000);
    loadPositionFile("../resources/other/E12.46FRC-1250k-D12-1s-Resolved.book", 10000000);
    auto K = tuning::computeK(2.86681, 200, 1e-7, 16);
    
    for (int i = 0; i < 5000; i++) {
        
        std::cout << "--------------------------------------------------- [" << i
                  << "] ----------------------------------------------" << std::endl;
        
        // std::cout << tuning::optimisePSTBlackBox(evaluator, K, pieceScores, 6, 1) << std::endl;
        // std::cout << tuning::optimisePSTBlackBox(evaluator, K, &evfeatures[21], 2, 1) << std::endl;
        // std::cout << tuning::optimisePSTBlackBox(evaluator, K, pinnedEval, 15, 1) << std::endl;
        // std::cout << tuning::optimisePSTBlackBox(evaluator, K, hangingEval, 5, 1) << std::endl;
        std::cout << tuning::optimisePSTBlackBox(K, &bishop_pawn_same_color_table_o[0], 9, 1, 16) << std::endl;
        std::cout << tuning::optimisePSTBlackBox(K, &bishop_pawn_same_color_table_e[0], 9, 1, 16) << std::endl;
        
        for (Square s = 0; s < 9; s++) {
            std::cout << "M(" << setw(5) << MgScore(bishop_pawn_same_color_table_o[s]) << "," << setw(5)
                      << EgScore(bishop_pawn_same_color_table_o[s]) << "), ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        
        for (Square s = 0; s < 9; s++) {
            std::cout << "M(" << setw(5) << MgScore(bishop_pawn_same_color_table_e[s]) << "," << setw(5)
                      << EgScore(bishop_pawn_same_color_table_e[s]) << "), ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        /*for (Square s = 0; s < 23; s++) {
            std::cout << "M(" << setw(5) << MgScore(*evfeatures[s]) << "," << setw(5) << EgScore(*evfeatures[s])
                      << "), ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
*/
        /*for (Square s = 0; s < 15; s++) {
            std::cout << "M(" << setw(5) << MgScore(pinnedEval[s]) << "," << setw(5) << EgScore(pinnedEval[s]) << "), ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for (Square s = 0; s < 5; s++) {
            std::cout << "M(" << setw(5) << MgScore(hangingEval[s]) << "," << setw(5) << EgScore(hangingEval[s])
                      << "), ";
            std::cout << std::endl;
        }
        std::cout << std::endl;*/
    }
    
    delete evaluator;
    bb_cleanUp();
}

#include "fecppnn/Data.h"
#include "fecppnn/Network.h"
#include "fecppnn/config.h"
#include "fecppnn/optimiser.h"
#include "fecppnn/sample.h"
#include "fecppnn/structure.h"


// TODO this should be in the nn namespace, probably
//      with a better name, I think
struct training_example {
    nn::Sample input;
    nn::Data target;
};

std::vector<training_example> generate_fake_data() {
    std::vector<training_example> v;
    
    for(int i = 0; i < 100000; i++) {
        nn::Sample input {};
        int idx = rand() % 12*64;
        input.indices.push_back(idx);
        
        nn::Data target {1, 0};
        target(0) = idx / (12 * 64.0);
        
        training_example e = {
            input,
            target
        };
        
        v.push_back(e);
    }
    
    return v;
}


std::vector<training_example> load_samples_from_file() {
    std::vector<training_example> v;
    
    for(int i = 0; i < 1000; i++) {
        nn::Sample input {};
        input.indices.push_back(0);
        
        nn::Data target {1, 0};
        target(0) = 1;
        
        training_example e = {
            input,
            target
        };
        
        
        v.push_back(e);
    }
    
    return v;
}

int main(int argc, char* argv[]) {
    
    int threadID = 0;

    nn::Network net {};
    std::vector<training_example> train_dataset = generate_fake_data();
    net.compute(&train_dataset[0].input, threadID);


    for(int n = 0; n < 1000; n++){

        float lossSum = 0;
        for (int i = 0; i < train_dataset.size(); i++) {
            net.compute(&train_dataset[i].input, threadID);
            float loss = net.applyLoss(&train_dataset[i].target, threadID);
            net.backprop(&train_dataset[i].input, threadID);
            lossSum += loss;

            if(i % NN_BATCH_SIZE == NN_BATCH_SIZE-1){
                net.optimise();
                net.mergeGrad();
                net.clearGrad();
            }
            
        }
        std::cout << "loss= " << lossSum << std::endl;
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
