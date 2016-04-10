#ifndef EightPointAlgorithm_h_PFJzKM4xWyWc
#define EightPointAlgorithm_h_PFJzKM4xWyWc

#include <Eigen/Dense>

Eigen::Matrix<double, 3, 3> EightPointAlgorithm(const Eigen::MatrixXd& matches);

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
	crossProductMatrix(2,1) = -paramVector(0);
	return crossProductMatrix;
}

/**
 * Reconstructs the camera projection matrix from the fundamental matrix F.
 */
Eigen::Matrix<double, 3, 3> reconstructCameraProjectionMatrix (const  Eigen::Matrix<double, 3, 3> &fundamentalMatrix) {
	// Initialise solver.
	const Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd(fundamentalMatrix.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

	// Initiliase right hand side.
	const Eigen::Matrix<double, 3, 1> rhs = Eigen::Matrix<double, 3, 1>::Zero();

	// Determine epipole
	const  Eigen::Matrix<double, 3, 1> epipole = svd.solve(rhs);

	// Compute camera projection matrix.
	const Eigen::Matrix<double, 3, 3> crossProductEpipole = crossProductMatrix(epipole);
	const Eigen::Matrix<double, 3, 3> cameraProjectionMatrix = - crossProductEpipole *fundamentalMatrix;

	return cameraProjectionMatrix;
}

} // anon namespace

#endif // EightPointAlgorithm_h_PFJzKM4xWyWc
