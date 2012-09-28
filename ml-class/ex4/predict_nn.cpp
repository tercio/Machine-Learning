/*
 *  Predict usando o Armadillo para C++
 *
 *   g++ -ggdb -o predict_nn predict_nn.cpp -O1 -larmadillo
 *
 *
 *
 */


#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

	/*
	 // exemplo
	mat A (2,2);

	A(0,0) = 2;
	A(0,1) = 3;
	A(1,0) = 4;
	A(1,1) = 5;

	//mat O = ones<mat> (2,1);
	//A.insert_cols(0,O);

	//cout << A << endl;

	//cout << exp(-A) << endl;
	// fim do exemplo
	*/

	mat Exemplo;
	mat Theta1, Theta2;

	Exemplo.load("/home/tercio/projects/tercio/Machine-Learning/ml-class/ex4/exemplo.mat",raw_ascii); 
	Theta1.load("/home/tercio/projects/tercio/Machine-Learning/ml-class/ex4/Theta1.mat",raw_ascii); 
	Theta2.load("/home/tercio/projects/tercio/Machine-Learning/ml-class/ex4/Theta2.mat",raw_ascii); 

	//cout << Exemplo;

	// sigmoid function
	// g = 1.0 ./ (1.0 + exp(-z));	


	int m=1;
	int num_labels = 10;


	Row<double> Ones = ones<vec> (1);

	Exemplo.insert_cols(0,Ones);
	Row<double> h1 = 1.0 / (1.0 + exp(-(Exemplo * trans(Theta1) )));
	//cout << "h1-> " << h1.n_rows << " x " << h1.n_cols<<endl;

	//cout << h1 << endl;


	h1.insert_cols(0,Ones);
	mat h2 = 1.0 / (1.0 + exp(-(h1 * trans(Theta2) )));
	//cout << "h2-> " << h2.n_rows << " x " << h2.n_cols<<endl;

	cout << h2<<endl;
	uword index;
   	double val = h2.max(index);

	index ++;
	cout << "ans: " << index % 10 << "  (index: " << index<< ") (val: " << val << ")" << endl;

	return 0;
}
