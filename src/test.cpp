#include "libs.h"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "EightPointAlgorithm.h"

#define NUM_POINTS 9
const double handmade_points[NUM_POINTS][6] = {
	// 6447,     6448       6451
	{ 466,  427,  362,  207,  504,   60}, // Ecke auf Fahne
	{1370,  613, 1593,  509, 1764,  320}, // Extruderspitze Deltadrucker
	{ 501,  796,  335,  572,  504,  449}, // linke Gewindestange Mendel
	{1010,  714,  966,  564, 1146,  421}, // rechte Gewindestange Mendel
	{1269,  323, 1595,  141,  NAN,  NAN}, // Fleck auf Fahne
	{ 876, 1216,  736, 1069,  899,  852}, // rechtes Druckobjekt Mendel
	{1551,  771, 1493,  732, 1673,  735}, // Griff Abkantbank
	{1676, 1266, 1505, 1394, 1600, 1265}, // Schraube Abkantbank
	{1271, 1125, 1357, 1091, 1459,  817}  // Fleck auf Tisch
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

void DrawEpipolarLines(
	const std::string& base_img,
	const std::string& target_img,
	size_t base_offset,
	size_t target_offset
)
{
	std::vector<size_t> used_rows;
	for (size_t row = 0; row < NUM_POINTS; row++ ) {
		if(
				std::isfinite(handmade_points[row][base_offset +0]) &&
				std::isfinite(handmade_points[row][base_offset +1]) &&
				std::isfinite(handmade_points[row][target_offset +0]) &&
				std::isfinite(handmade_points[row][target_offset +1])
		)
			used_rows.push_back(row);
	};
	Eigen::MatrixXd points = Eigen::MatrixXd(used_rows.size(), 4);
	for(size_t i = 0; i < used_rows.size(); ++i) {
		size_t row = used_rows.at(i);
		points(i, 0) = handmade_points[row][base_offset +0];
		points(i, 1) = handmade_points[row][base_offset +1];
		points(i, 2) = handmade_points[row][target_offset +0];
		points(i, 3) = handmade_points[row][target_offset +1];
	}

	auto fundamental = EightPointAlgorithm(points);
	std::cout << "Here are the points:" << std::endl << points<< std::endl;
	std::cout << "TFT = [\n" << fundamental << "]\n";
	std::cout << "% Input errors: ";

	for (int row = 0; row < points.rows(); row++ ) {
		Eigen::Matrix<double, 3, 1> left, right;
		left(0) = points(row, 0);
		left(1) = points(row, 1);
		left(2) = 1;
		right(0) = points(row, 2);
		right(1) = points(row, 3);
		right(2) = 1;
		std::cout << (right.transpose() * fundamental * left);
		std::cout << (row +1 < points.rows() ? ", " : "\n");
	}

	Eigen::Matrix<double, 3, 3> A;
	Eigen::Matrix<double, 3, 1> b;
	std::tie(A, b) = ReconstructCameraProjectionMatrix(fundamental);
	std::cout << "A = [\n" << A << "\n]\n";
	std::cout << "b = [\n" << b << "\n]\n";

	// -- Debug output --
	std::string filename = "/tmp/" + target_img + "_from_" + base_img + ".svg";
	std::ofstream svg(filename);
	svg <<
		"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
		"<svg width=\"2048\" height=\"1536\" "
		"xmlns=\"http://www.w3.org/2000/svg\" "
		"xmlns:xlink=\"http://www.w3.org/1999/xlink\" >\n"
		"<image xlink:href=\"IMG_" << target_img << ".JPG\" />\n"
	;
	for (size_t row = 0; row < points.rows(); row++ ) {
		Eigen::Matrix<double, 3, 1> anchor;
		anchor << points(row, 0), points(row, 1), 1.0;
		double z = 0.1;
		Eigen::Matrix<double, 3, 1> p1 = -0.04 * A * anchor + b;
		Eigen::Matrix<double, 3, 1> p2 = 0.04 * A * anchor + b;
		p1 *= (1.0 / p1(2));
		p2 *= (1.0 / p2(2));
		svg << "<line x1=\"" << p1(0) << "\" y1=\"" << p1(1);
		svg << "\" x2=\"" << p2(0) << "\" y2=\"" << p2(1);
		svg << "\" style=\"stroke:";
		switch (row & 3) {
		case 0:  svg << "#900"; break;
		case 1:  svg << "#080"; break;
		case 2:  svg << "#00b"; break;
		default: svg << "#760"; break;
		};
		svg <<";stroke-width:11\" />\n";
	};
	for (size_t row = 0; row < points.rows(); row++ ) {
		svg << "<circle r=\"21\" style=\"fill: ";
		switch (row & 3) {
		case 0:  svg << "#f22"; break;
		case 1:  svg << "#1e1"; break;
		case 2:  svg << "#44f"; break;
		default: svg << "#de0"; break;
		};
		svg << "; stroke: #800\" cx=\"" << points(row, 2);
		svg << "\" cy=\"" << points(row, 3) << "\" />\n";
	};
	svg << "</svg>\n";
	svg.close();
}

} // anon namespace


int main (void) {
	DrawEpipolarLines("6447", "6448", 0, 2);
	DrawEpipolarLines("6448", "6451", 2, 4);
	DrawEpipolarLines("6451", "6447", 4, 0);

	DrawEpipolarLines("6448", "6447", 2, 0);
	DrawEpipolarLines("6451", "6448", 4, 2);
	DrawEpipolarLines("6447", "6451", 0, 4);
}
