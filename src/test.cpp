#include "libs.h"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "EightPointAlgorithm.h"

#define NUM_POINTS 9
const double handmade_points[NUM_POINTS][4] = {
	{ 466,  427,  362,  207}, // Ecke auf Fahne
	{1370,  613, 1593,  509}, // Extruderspitze Deltadrucker
	{ 501,  796,  335,  572}, // linke Gewindestange Mendel
	{1010,  714,  966,  564}, // rechte Gewindestange Mendel
	{1269,  323, 1595,  141}, // Fleck auf Fahne
	{ 876, 1216,  736, 1069}, // rechtes Druckobjekt Mendel
	{1551,  771, 1493,  732}, // Griff Abkantbank
	{1676, 1266, 1505, 1394}, // Schraube Abkantbank
	{1271, 1125, 1357, 1091}  // Fleck auf Tisch
};


namespace {

/**
 * Computes a matrix whose left-multiplication is equivalent to computing the cross product with the given vector.
 */
Eigen::Matrix< double , 3 , 3> crossProductMatrix(const Eigen::Matrix< double , 3 , 1> &paramVector) {
	Eigen::Matrix< double , 3 , 3> crossProductMatrix = Eigen::Matrix<double, 3, 3>().setZero();
	crossProductMatrix(0,1) = -paramVector(2);
	crossProductMatrix(0,2) = paramVector(1);
	crossProductMatrix(1,0) = paramVector(2);
	crossProductMatrix(1,2) = -paramVector(0);
	crossProductMatrix(2,0) = -paramVector(1);
	crossProductMatrix(2,1) = paramVector(0);
	return crossProductMatrix;
}

/**
 * Reconstructs the camera projection matrix from the fundamental matrix F.
 * @return (A, b)
 */
std::tuple<
	Eigen::Matrix<double, 3, 3>,
	Eigen::Matrix<double, 3, 1>
>
ReconstructCameraProjectionMatrix (const  Eigen::Matrix<double, 3, 3> &fundamentalMatrix) {
	// Initialise solver.
	Eigen::Matrix<double, 4, 3> to_solve;
	to_solve.block<3, 3>(0, 0) = fundamentalMatrix.transpose();
	to_solve.block<1, 3>(3, 0) << 0, 0, 1;
	const Eigen::JacobiSVD<Eigen::Matrix<double, 4, 3>> svd(to_solve, Eigen::ComputeFullU | Eigen::ComputeFullV);

	// Initiliase right hand side.
	Eigen::Matrix<double, 4, 1> rhs;
	rhs << 0, 0, 0, 1;

	// Determine epipole
	const  Eigen::Matrix<double, 3, 1> epipole = svd.solve(rhs);

	// Compute camera projection matrix.
	const Eigen::Matrix<double, 3, 3> crossProductEpipole = crossProductMatrix(epipole);
	const Eigen::Matrix<double, 3, 3> cameraProjectionMatrix = - crossProductEpipole *fundamentalMatrix;
	//const Eigen::Matrix<double, 3, 3> cameraProjectionMatrix = - epipole.cross(fundamentalMatrix);

	return std::make_tuple(cameraProjectionMatrix, epipole);
}

} // anon namespace


int main (void) {
	Eigen::MatrixXd points = Eigen::MatrixXd(NUM_POINTS, 4);
	for (int row = 0; row < NUM_POINTS; row++ ) {
		//for (int col = 0; col < 4; col++) {
		//	points(row, col) = handmade_points[row][col];
		//}
		points(row, 0) = handmade_points[row][2];
		points(row, 1) = handmade_points[row][3];
		points(row, 2) = handmade_points[row][0];
		points(row, 3) = handmade_points[row][1];
	}
	auto fundamental = EightPointAlgorithm(points);
	std::cout << "Here are the points:" << std::endl << points<< std::endl;
	std::cout << "TFT = [\n" << fundamental << "]\n";
	std::cout << "% Input errors: ";

	for (int row = 0; row < NUM_POINTS; row++ ) {
		Eigen::Matrix<double, 3, 1> left, right;
		left(0) = points(row, 0);
		left(1) = points(row, 1);
		left(2) = 1;
		right(0) = points(row, 2);
		right(1) = points(row, 3);
		right(2) = 1;
		std::cout << (right.transpose() * fundamental * left);
		std::cout << (row +1 < NUM_POINTS ? ", " : "\n");
	}

	Eigen::Matrix<double, 3, 3> A;
	Eigen::Matrix<double, 3, 1> b;
	std::tie(A, b) = ReconstructCameraProjectionMatrix(fundamental);
	std::cout << "A = [\n" << A << "\n]\n";
	std::cout << "b = [\n" << b << "\n]\n";

	// -- Debug output --
	std::ofstream left_svg("/tmp/left.svg");
	left_svg << (
		"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
		"<svg width=\"2048\" height=\"1536\" "
		"xmlns=\"http://www.w3.org/2000/svg\" "
		"xmlns:xlink=\"http://www.w3.org/1999/xlink\" >\n"
		"<image xlink:href=\"IMG_6448.JPG\" />\n"
	);
	for (size_t row = 0; row < NUM_POINTS; row++ ) {
		left_svg << "<circle r=\"9\" style=\"fill: #1e1; stroke: #800\" cx=\"" << points(row, 0);
		left_svg << "\" cy=\"" << points(row, 1) << "\" />\n";
	};
	left_svg << "</svg>\n";
	left_svg.close();

	std::ofstream right_svg("/tmp/right.svg");
	right_svg << (
		"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
		"<svg width=\"2048\" height=\"1536\" "
		"xmlns=\"http://www.w3.org/2000/svg\" "
		"xmlns:xlink=\"http://www.w3.org/1999/xlink\" >\n"
		"<image xlink:href=\"IMG_6447.JPG\" />\n"
	);
	for (size_t row = 0; row < NUM_POINTS; row++ ) {
		Eigen::Matrix<double, 3, 1> anchor;
		anchor << points(row, 0), points(row, 1), 1.0;
		double z = 0.1;
		Eigen::Matrix<double, 3, 1> p1 = 0.4 * A * anchor + b;
		Eigen::Matrix<double, 3, 1> p2 = 0.04 * A * anchor + b;
		p1 *= (1.0 / p1(2));
		p2 *= (1.0 / p2(2));
		right_svg << "<line x1=\"" << p1(0) << "\" y1=\"" << p1(1);
		right_svg << "\" x2=\"" << p2(0) << "\" y2=\"" << p2(1);
		right_svg << "\" style=\"stroke:#008000;stroke-width:11\" />\n";

		right_svg << "<circle r=\"15\" style=\"fill: #1e1; stroke: #800\" cx=\"" << points(row, 2);
		right_svg << "\" cy=\"" << points(row, 3) << "\" />\n";
	};
	right_svg << "</svg>\n";
	right_svg.close();

}
