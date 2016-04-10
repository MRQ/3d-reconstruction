#include <iostream>
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


int main (void) {
	Eigen::MatrixXd points = Eigen::MatrixXd(NUM_POINTS, 4);
	for (int row = 0; row < NUM_POINTS; row++ ) {
		for (int col = 0; col < 4; col++) {
			points(row, col) = handmade_points[row][col];
		}
	}
	auto projection = EightPointAlgorithm(points);
	std::cout << "Here are the points:" << std::endl << points<< std::endl;
	std::cout << "Here is the projection F:" << std::endl << projection << std::endl;
}
