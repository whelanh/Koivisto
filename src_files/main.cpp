
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

int main(int argc, char* argv[]) {

    //    startMeasure();
    //    for(int i = 0; i < 2e6; i++){
    //        NETWORK->compute();
    //    }
    //    std::cout << stopMeasure() << std::endl;

    nn::Network net {};
    nn::Sample  sample1 {};
    for (int i = 0; i < 1; i++) {
        sample1.indices.push_back(i);
    }
    nn::Data target {1, 0};
    target(0) = 1;

    startMeasure();
    int threadID = 1;

    for (int i = 0; i < 100000; i++) {
        net.compute(&sample1, threadID);
        std::cout << net.applyLoss(&target, threadID) << std::endl;
        net.backprop(&sample1, threadID);

        net.mergeGrad();
        net.optimise();
        net.clearGrad();
    }
    std::cout << stopMeasure() << std::endl;

    net.compute(&sample1, threadID);
    std::cout << net.getOutput(threadID) << std::endl;

    /**********************************************************************************
     *                                  T U N I N G                                   *
     **********************************************************************************/

    // main_tune_pst_bb(PAWN);
    //    eval_init();
    //     main_tune_features();
    // main_tune_pst();
    // main_tune_features_bb();

    return 0;
}
