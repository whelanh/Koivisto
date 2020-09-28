
#include "Bitboard.h"
#include "Board.h"
#include "Move.h"
#include "MoveOrderer.h"
#include "Tuning.h"
#include "Verification.h"
#include "fecppnn/network/Network.h"
#include "fecppnn/special/KingPieceNetwork.h"
#include "uci.h"

#include <iomanip>

using namespace std;
using namespace bb;
using namespace move;

//#include "fecppnn/network/Network.h"
//#include "fecppnn/special/KingPieceNetwork.h"

int main(int argc, char* argv[]) {

    if (argc == 1) {
        uci_loop(false);
    } else if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        uci_loop(true);
    }

//        bb_init();
//
//        fecppnn::Network*         net = fecppnn::createNetwork("net1.structure");
//        fecppnn::KingPieceNetwork worker {net};
//
//        net->writeNetworkStructure("net2.structure");
//
//        //    worker.getNetwork()->loadWeights("nn.bin");
//
//        Board b {"1k6/8/8/1P4rP/2P3P1/q7/P3KR2/7R w - - 0 55"};
//        worker.resetInput(&b);
//        for (int i = 0; i < worker.getWhiteInput()->getInputTracker().count(); i++) {
//            std::cout << worker.getWhiteInput()->getInputTracker().at(i) << " ";
//        }
//        std::cout << std::endl;
//        for (int i = 0; i < worker.getBlackInput()->getInputTracker().count(); i++) {
//            std::cout << worker.getBlackInput()->getInputTracker().at(i) + 10*64*64 << " ";
//        }

    return 0;
}
