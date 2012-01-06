#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>

#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "util.hpp"


using namespace std;

// namespace alias
namespace boost_ublas = boost::numeric::ublas;


bool debug = true;

// The simple implememtation of hidden Markov Models.
class HMM {
    public:
        void init(const int n, const int m);
	void init_with_random(const int n, const int m);
        void set_seq_num(const int t) { T = t; }

        // estimate (A, B, PI) given the initial values
        void estimate_hmm(const boost_ublas::vector<int>& O);

        void reset() {
            PI.clear();
            A.clear();
            B.clear();
        }
        
        // discover the hidden state sequence that was most likely to have
        // produced a given observation sequence.
        void viterbi(const boost_ublas::vector<int>& O, boost_ublas::vector<int>& S);

    private:
        // The number of states
        int N; 
        // The number of the Observation symbols
        int M; 

        // The initial probability for each state
        boost_ublas::vector<double> PI;

        // The N * N State transition matrix
        boost_ublas::matrix<double> A;

        // The N * M Observation matrix
        boost_ublas::matrix<double> B;

        // the sequence num
        int T;

};

int generate_k_random_integers(std::vector<int>& s, int upper_bound, int k) {
    s.clear();
    s.reserve(k);
	
    int sum = 0;
    boost::mt19937 rng;   
    boost::uniform_int<> g(1, upper_bound);
    for (size_t i=0; i<(size_t)k; ++i) {
        int x = g(rng);
	sum += x;
        s.push_back(x);
    }
	
    return sum;
}


////////////////////////////The implementation of HMM
void HMM::init_with_random(const int n, const int m) {
    N = n;
    M = m;
    std::vector<int> s;
    int factor = 5;
    int sum = generate_k_random_integers(s, N * factor, N);

    PI.resize(N);
    for (int i=0; i<N; ++i) {
        PI(i) = s[i] / (double) sum;
    }
    cout << "---PI---" << endl;
    cout << PI << endl;

	
    A.resize(N, N);
    for (size_t i=0; i<A.size1(); ++i) {
        // for each row, generate random number
        int sum = generate_k_random_integers(s, N * factor, A.size2());
        for (size_t j=0; j<A.size2(); ++j) {
            A(i, j) = s[j] / (double) sum;
        }
    }
    cout << "---A---" << endl;
    cout << A << endl;

    B.resize(N, M);
    for (size_t i=0; i<B.size1(); ++i) {
        // for each row, generate random number
        int sum = generate_k_random_integers(s, M * factor, B.size2());
        for (size_t j=0; j<B.size2(); ++j) {
            B(i, j) = s[j] / (double) sum;
        }
    }
    cout << "---B---" << endl;
    cout << B << endl;
}

// fill A, B and PI with uniform distribution
void HMM::init(const int n, const int m) {
    N = n;
    M = m;

    PI.resize(N);
    for (size_t i=0; i<PI.size(); ++i) {
        PI(i) = 1.0 / (double) N;
    }
    cout << "---PI---" << endl;
    cout << PI << endl;

    A.resize(N, N);
    for (size_t i=0; i<A.size1(); ++i) {
        for (size_t j=0; j<A.size2(); ++j) {
            A(i, j) = 1.0 / (double) N;
        }
    }
    cout << "---A---" << endl;
    cout << A << endl;

    B.resize(N, M);
    for (size_t i=0; i<B.size1(); ++i) {
        for (size_t j=0; j<B.size2(); ++j) {
            B(i, j) = 1.0 / (double) M;
        }
    }
    cout << "---B---" << endl;
    cout << B << endl;
}

// the viterbi algorithm is very similar to the forward algorithm,
// except that the transition are maximised at each step, instead of summed
// R(t,i) = Max P(q1q2...qt=si,o1o2...ot | x) in which x is the HMM parameters
void HMM::viterbi(const boost_ublas::vector<int>& O, boost_ublas::vector<int>& S) {
    // step 1: Initialisation
    // for backtrace
    boost_ublas::matrix<int> X(T,N);
	boost_ublas::matrix<double> R(T,N);
    for (int i=0; i<T; ++i) {
        for (int j=0; j<N; ++j) {
            X(i,j) = 0;
            R(i,j) = 0;
        }
    }
    
    for (int i=0; i<N; ++i) {
        R(0,i) =-log(PI(i))-log(B(i,O(0)));
        X(0,i) = 0;
    }

    // step 2: Recursion
    for (int t=1; t<T; t++) {
        for (int i=0; i<N; ++i) {
            double max = -9999;
            int index = -1;
            for (int j=0; j<N; ++j) {
               double tmp =  -(log(R(t-1,j)) + log(A(j,i)) + log(B(i, O(t))));
               if (tmp >= max) {
                   max = tmp;
                   index = j;
               }
            }
            if (index == -1) {
                cout << "Recursion error!" << endl;
            }

            R(t,i) = max;
            X(t,i) = index;
        }
    }

    // step 3: termination
    int last_state_index = -1;
    double tmp = -9;
    for(int i=0; i<N; ++i) {
        if (tmp < R(T-1, i)) {
            tmp = R(T-1, i);
            last_state_index = i;
        }
    }


    // backtrace
    if (last_state_index == -1) {
        cout << "last index error..." << endl;
    }

    S.resize(T);
    int index = last_state_index;
    int pos = T - 1;
    S(pos) = index;
    
    for (int t=T-2; t>=0; --t) {
        index = X(t, index);
        pos--;
        S(pos) = index;
    }
}


// Please refer to paper: A revealing introduction to Hidden Markov 
// Models for details 
void HMM::estimate_hmm(const boost_ublas::vector<int>& O) {
    // step 1: initialization
    int max_iters = 100;
    int iters = 0;
    double oldLogProb = -999999; // give a very small value
    // T = |O|  
    set_seq_num(O.size());
    // the forward prob
    boost_ublas::matrix<double> a(T, N);
    // the scale factors 
    boost_ublas::vector<double> c(T);
    // the backward prob
    boost_ublas::matrix<double> b(T, N);
    // r(t)(i,j): the prob of being in state i at the time t and in state j at time t+1
    // R(t)(i): the prob of being in state i at time t, given the observation sequence
    boost_ublas::vector<boost_ublas::matrix<double> > r(T-1);
    boost_ublas::matrix<double> R(T-1,N);
	
    while (true) {
        // step 2: compute a(t)(i)
        // step 2.1 compute a(0)(i) for each i in [0, N-1]
        c(0) = 0;
        for (int i=0; i<N; ++i) {
            a(0, i) = PI(i) * B(i, O(0));
            c(0) += a(0,i);
        }
        // step 2.2 scale the a(0)
        c(0) = 1 / c(0);
        for (int i=0; i<N; ++i) {
            a(0,i) = c(0) * a(0,i);
        }

        // step 2.3 compute a(t)(i) for 1<=t<T and 0<=i<N
        // a(t)(i) = { sum(a(t-1)(j)* A(j)(i) } * B(i)(O(t)) for j in (0, N) and t in (1, T)
        for (int t=1; t<T; t++) {
            c(t) = 0;
            for (int i=0; i<N; ++i) {
                a(t, i) = 0;
                for (int j=0; j<N; ++j) {
                    a(t,i) += a(t-1,j) * A(j,i);
                }
                a(t, i) = a(t, i) * B(i, O(t));
                c(t) += a(t, i);
            }

            // step 2.4 scale a(t)(i)
            c(t) = 1/c(t);
            for (int i=0; i<N; ++i) {
                a(t,i) = c(t) * a(t,i);
            }
        }

        // step 3: compute backward prob, i.e., b(t)(i)
        for (int i=0; i<N; i++) {
            b(T-1, i) = c(T-1);
        }
        for (int t=T-2; t>=0; --t) {
            for (int i=0; i<N; ++i) {
                b(t,i) = 0;
                for (int j=0; j<N; ++j) {
                    b(t,i) += A(i,j) * B(j, O(t+1)) * b(t+1, j);
                    // scale with the same scale factor as a(t,i)
                    b(t,i) = c(t) * b(t, i);
                }
            }
        }


        // step 4: compute r(t)(i,j) and R(t)(i)
        for (int t=0; t<T-1; ++t) {
            r(t).resize(N, N);
        }
        for (int t=0; t<T-1; t++) {
            double denom = 0;
            for (int i=0; i<N; i++) {
                for (int j=0; j<N; j++) {
                    denom += a(t, i) * A(i,j) * B(j,O(t+1)) * b(t+1,j);
                }
            }

            for (int i=0; i<N; ++i) {
                R(t,i)=0;
                for(int j=0; j<N; ++j) {
                    r(t)(i, j) = (a(t,i) * A(i,j) * B(j,O(t+1)) * b(t+1, j)) / denom;
                    R(t,i) += r(t)(i,j);
                }
            }
        }


        // step 5: re-estimate A, B, and PI
        // step 5.1: re-estimate PI
        for (int i=0; i<N; ++i) {
            PI(i) = R(0,i);
        }

        // step 5.2: re-estimate A
        for (int i=0; i<N; ++i) {
            for (int j=0; j<N; ++j) {
                double numer = 0;
                double denom = 0;
                for (int t=0; t<T-1; ++t) {
                    numer += r(t)(i,j);
                    denom += R(t,i);
                }
                A(i,j) = numer / denom;
            }
        }

        // step5.3 re-estimate B
        for (int i=0; i<N; ++i) {
            for (int j=0; j<M; ++j) {
                double numer = 0;
                double denom = 0;
                for (int t=0; t<T-1; t++) {
                    if (O(t) == j) {
                        numer += R(t,i);
                    }
                    denom += R(t, i);
                }

                B(i,j) = numer / denom;
            }
        }

        // step 6: compute log[P(O|x)]
        double logProb = 0;
        for (int i=0; i<T; ++i) {
            logProb += log(c(i));
        }
        logProb = -logProb;
        iters += 1;
        if (iters < max_iters && logProb > oldLogProb) {
            oldLogProb = logProb;
        }
        else {
            // stop here
            cout << "the final values: << ";
            cout << " A-----" << endl;
            cout << A << endl;

            cout << "---PI---" << endl;
            cout << PI << endl;

            cout << "---B---" << endl;
            cout << B << endl;
            break;
        }
    }
}


/////////////////////////// Test
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " N M" << endl;
        return -1;
    }

    int N = 0;
    int M = 0;
    convert_from_string(N, argv[1]);
    convert_from_string(M, argv[2]); 

    cout << "Number of states:" << N << endl;
    cout << "Number of observation symbols:" << M << endl;
    
    HMM hmm;
    hmm.init(N, M);

    boost_ublas::vector<int> O(M * 2);
    boost::mt19937 rng;   
    boost::uniform_int<> g(0, M-1);
    for (int i=0; i<M * 2; ++i) {
        O(i) = g(rng);
    }
    cout << "Observation sequence...." << endl;
    cout << O << endl;

    // case 1: test estimate parameters for HMM given observation sequences
    hmm.estimate_hmm(O);

    // case 2: test Viterbi algorithm
    boost_ublas::vector<int> S;
    hmm.reset();
    hmm.init_with_random(N, M);
    hmm.viterbi(O,S);
    cout << "The best state sequence is:" << endl;
    cout << S << endl;

    return 0;
}
