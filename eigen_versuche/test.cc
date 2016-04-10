#include <iostream>
#include <Eigen/Dense>

#define NUM_POINTS 8

Eigen::MatrixXf eightPointMethod (const Eigen::MatrixXf paramPoints);

const double handmade_points[NUM_POINTS][4] = {
	{ 466,  427,  362,  207}, // Ecke auf Fahne
	{1370,  613, 1593,  509}, // Extruderspitze Deltadrucker
	{ 501,  796,  335,  572}, // linke Gewindestange Mendel
	{1010,  714,  966,  564}, // rechte Gewindestange Mendel
	{1269,  323, 1595,  141}, // Fleck auf Fahne
	{ 876, 1216,  736, 1069}, // rechtes Druckobjekt Mendel
	{1551,  771, 1493,  732}, // Griff Abkantbank
	{1676, 1266, 1505, 1394}, // Schraube Abkantbank
	//{1271, 1125, 1357, 1091}  // Fleck auf Tisch
};

void normaliseInput(const double rawData[NUM_POINTS][4], Eigen::MatrixXf &normalisedMatrix) {
	const double sqrtTwo = std::sqrt(2.0);
	normalisedMatrix = Eigen::MatrixXf(NUM_POINTS, 4);
	double mx, my, xscale, yscale, tmp;

	mx = 0.0d;
	my = 0.0d;;
	for (int row = 0; row < NUM_POINTS; row++ ) {
		mx += handmade_points[row][0] +handmade_points[row][2];
		my += handmade_points[row][1] +handmade_points[row][3];
	}
	mx = 0.5 * (mx / NUM_POINTS);
	my = 0.5 * (my / NUM_POINTS);

	xscale = 0.0d;
	yscale = 0.0d;
	for (int row = 0; row < NUM_POINTS; row++ ) {
		tmp = (handmade_points[row][0] - mx);
		xscale += tmp*tmp;
		tmp = (handmade_points[row][2] - mx);
		xscale += tmp*tmp;

		tmp = (handmade_points[row][0] - my);
		yscale += tmp*tmp;
		tmp = (handmade_points[row][2] - my);
		yscale += tmp*tmp;
	}

	xscale = sqrtTwo / std::sqrt(0.5 * xscale / NUM_POINTS);
	yscale = sqrtTwo / std::sqrt(0.5 * yscale / NUM_POINTS);

	for (int row = 0; row < NUM_POINTS; row++ ) {
		normalisedMatrix(row, 0) = (handmade_points[row][0] - mx) * xscale;
		normalisedMatrix(row, 2) = (handmade_points[row][2] - mx) * xscale;
		normalisedMatrix(row, 1) = (handmade_points[row][1] - my) * yscale;
		normalisedMatrix(row, 3) = (handmade_points[row][3] - my) * yscale;
	}
}

int main (void) {
	Eigen::MatrixXf points = Eigen::MatrixXf(NUM_POINTS, 4);
	const double scale = 0.001;
	for (int row = 0; row < NUM_POINTS; row++ ) {
		for (int col = 0; col < 4; col++) {
			points(row, col) = handmade_points[row][col] * scale;
		}
	}
	/*Eigen::MatrixXf points;
	normaliseInput(handmade_points, points); */

	Eigen::MatrixXf projection = eightPointMethod(points);
	std::cout << "Here are the points:" << std::endl << points<< std::endl;
	std::cout << "Here is the projection F:" << std::endl << projection << std::endl;
}

Eigen::MatrixXf eightPointMethod (const Eigen::MatrixXf paramPoints) {
	const int numberColumns = 9;
	Eigen::MatrixXf matrix = Eigen::MatrixXf(paramPoints.rows() +1, numberColumns);
	for (int row = 0; row < paramPoints.rows(); row++) {
		matrix(row, 0) = paramPoints(row, 2) * paramPoints(row, 0); // u' * u
		matrix(row, 1) = paramPoints(row, 2) * paramPoints(row, 1); // u' * v
		matrix(row, 2) = paramPoints(row, 2); // u'
		matrix(row, 3) = paramPoints(row, 3) * paramPoints(row, 0); // u * v'
		matrix(row, 4) = paramPoints(row, 3) * paramPoints(row, 1); // v * v'
		matrix(row, 5) = paramPoints(row, 3); // v'
		matrix(row, 6) = paramPoints(row, 0); // u
		matrix(row, 7) = paramPoints(row, 1); // v
		matrix(row, 8) = 1.0;
	}
	matrix.block<1, 8>(paramPoints.rows(), 0).setZero();
	matrix(paramPoints.rows(), 8) = 1;

	std::cout << "matrix 'matrix':\n" << matrix << '\n';

	// Initialise solver.
	const Eigen::JacobiSVD<Eigen::MatrixXf> svd(matrix,  Eigen::ComputeFullU | Eigen::ComputeFullV);

	// Initiliase right hand side.
	Eigen::VectorXf rhs = Eigen::VectorXf::Zero(matrix.rows());
	rhs(matrix.rows() -1) = 1;

	// Solve via SVD
	const  Eigen::MatrixXf parameters = svd.solve(rhs);
	std::cout << "matrix 'parameters':\n" << parameters << '\n';
	std::cout << "Probe:\n" << (matrix * parameters) << "\n";

	// Write out result.
	Eigen::MatrixXf result(3, 3);

	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			result(row, col) = parameters(3*row + col);
		}
	}
	return result;
};
