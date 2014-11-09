#include "HMMlib/hmm_table.hpp"
#include "HMMlib/hmm_vector.hpp"
#include "HMMlib/hmm.hpp"
using namespace hmmlib;

#include <iostream>

int main(int argc, char *args[]) {
    int K = 2; // number of states
    int M = 2; // alphabet size
    int n = 4; // length of observed sequence

    boost::shared_ptr<HMMVector<double> > pi_ptr(new HMMVector<double>(K));
    boost::shared_ptr<HMMMatrix<double> > T_ptr(new HMMMatrix<double>(K, K));
    boost::shared_ptr<HMMMatrix<double> > E_ptr(new HMMMatrix<double>(M, K));

    HMMVector<double> &pi = *pi_ptr;
    // initial probabilities
    pi(0) = 0.2; pi(1) = 0.8;

    HMMMatrix<double> &T = *T_ptr;
    // transitions from state 0
    T(0, 0) = 0.1; T(0, 1) = 0.9;
    // transitions from state 1
    T(1, 0) = 0.9; T(1, 1) = 0.1;

    HMMMatrix<double> &E = *E_ptr;
    // emissions from state 0
    E(0, 0) = 0.25; E(1, 0) = 0.75;
    // emissions from state 1
    E(0, 1) = 0.75; E(1, 1) = 0.25;

    std::cout << "Constructing HMM" << std::endl;
    HMM<double> hmm(pi_ptr, T_ptr, E_ptr);

    std::cout << "obs : [0, 1, 0, 1]" << std::endl;
    sequence obs(n);
    obs[0] = 0;
    obs[1] = 1;
    obs[2] = 0;
    obs[3] = 1;
    std::cout << "obs length: " << obs.size() << std::endl;

    std::cout << "Running viterbi" << std::endl;
    sequence hiddenseq(n);
    double loglik = hmm.viterbi(obs, hiddenseq);
    std::cout << "-- hiddenseq: [" << hiddenseq[0] << ", " << hiddenseq[1] << ", " << hiddenseq[2] << ", " << hiddenseq[3] << "]" << std::endl;
    std::cout << "-- log likelihood of hiddenseq: " << loglik << std::endl;

    std::cout << "Running forward" << std::endl;
    HMMMatrix<double> F(n, K);
    HMMVector<double> scales(n);
    hmm.forward(obs, scales, F);

    std::cout << "Running likelihood" << std::endl;
    loglik = hmm.likelihood(scales);
    std::cout << "-- loglikelihood of obs: " << loglik << std::endl;

    std::cout << "Running backward" << std::endl;
    HMMMatrix<double> B(n, K);
    hmm.backward(obs, scales, B);

    std::cout << "Running posterior decoding" << std::endl;
    HMMMatrix<double> pd(n, K);
    hmm.posterior_decoding(obs, F, B, scales, pd);

    std::cout << "Running Baum-Welch" << std::endl;
    boost::shared_ptr<HMMVector<double> > pi_counts_ptr(new HMMVector<double>(K));
    boost::shared_ptr<HMMMatrix<double> > T_counts_ptr(new HMMMatrix<double>(K, K));
    boost::shared_ptr<HMMMatrix<double> > E_counts_ptr(new HMMMatrix<double>(M, K));
    hmm.baum_welch(obs, F, B, scales, *pi_counts_ptr, *T_counts_ptr, *E_counts_ptr);

    std::cout << "Constructing new HMM" << std::endl;
    HMM<double> hmm2(pi_counts_ptr, T_counts_ptr, E_counts_ptr);

    std::cout << "Running forward on new HMM" << std::endl;
    hmm2.forward(obs, scales, F);
    std::cout << "Running likelihood on new HMM" << std::endl;
    loglik = hmm2.likelihood(scales);
    std::cout << "-- loglikelihood of obs in new HMM: " << loglik << std::endl;

    system("pause");
}