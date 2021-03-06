#include "libs.h"
#include "EightPointAlgorithm.h"

#include <tuple>
#include <iostream>

namespace {

/// @return normalized (normalized_matched, input_scaling)
std::tuple<
	Eigen::Matrix<double, Eigen::Dynamic, 4>,
	Eigen::Matrix<double, 3, 3>
>
NormalizeInput(const Eigen::MatrixXd& matches)
{
	using namespace Eigen;

	Eigen::Matrix<double, Eigen::Dynamic, 4>
		normalised_matrix (matches.rows(), 4);

	// - Means -
	Eigen::Matrix<double, 1, 2> sum;
	for (size_t row = 0; row < matches.rows(); row++) {
		sum += matches.block<1, 2>(row, 0);
		sum += matches.block<1, 2>(row, 2);
	}
	const Eigen::Matrix<double, 1, 2> mean = sum * (0.5/matches.rows());

	Eigen::Matrix<double, 1, 2> moment_1_sum;
	moment_1_sum.setZero();
	for (size_t row = 0; row < matches.rows(); row++) {
		auto shifted1 = matches.block<1, 2>(row, 0) - mean;
		moment_1_sum += shifted1 * shifted1.asDiagonal();

		auto shifted2 = matches.block<1, 2>(row, 2) - mean;
		moment_1_sum += shifted2 * shifted2.asDiagonal();
	}

	// faktor sqrt(2) implicit.
	const Eigen::Matrix<double, 1, 2>
		scale = (moment_1_sum * (1.0 / matches.rows())).cwiseSqrt().cwiseInverse();

	Eigen::Matrix<double, Eigen::Dynamic, 4> normalized(matches.rows(), 4);
	for (size_t row = 0; row < matches.rows(); row++ ) {
		normalized.block<1, 2>(row, 0) =
			(matches.block<1, 2>(row, 0) - mean) * scale.asDiagonal();
		normalized.block<1, 2>(row, 2) =
			(matches.block<1, 2>(row, 2) - mean) * scale.asDiagonal();
	}

	Eigen::Matrix<double, 3, 3> input_scaling;
	input_scaling.block<2, 2>(0, 0) = scale.asDiagonal();
	input_scaling.block<2, 1>(0, 2) = -mean * scale.asDiagonal();
	input_scaling.block<1, 2>(2, 0).setZero();
	input_scaling(2, 2) = 1;
	std::cout << "\n-- normalized: --\n" << normalized << "\n-- input_scaling --\n";
	std::cout << "\n----\n" << input_scaling << "\n----\n";

	return std::make_tuple(normalized, input_scaling);
}

} // anon namespace

Eigen::Matrix<double, 3, 3> EightPointAlgorithm(const Eigen::MatrixXd& matches)
{
	using namespace Eigen;

	Matrix<double, Dynamic, 4> normalized;
	Matrix<double, 3, 3> input_scaling;

	std::tie(normalized, input_scaling) = NormalizeInput(matches);

	Matrix<double, Dynamic, 9> A(normalized.rows() +1, 9);
	for(size_t row = 0; row < normalized.rows(); row++) {
		A(row, 0) = normalized(row, 2) * normalized(row, 0); // u' * u
		A(row, 1) = normalized(row, 2) * normalized(row, 1); // u' * v
		A(row, 2) = normalized(row, 2);                   // u'
		A(row, 3) = normalized(row, 3) * normalized(row, 0); // v' * u
		A(row, 4) = normalized(row, 3) * normalized(row, 1); // v' * v
		A(row, 5) = normalized(row, 3);                   // v'
		A(row, 6) =                     normalized(row, 0); // u
		A(row, 7) =                     normalized(row, 1); // v
		A(row, 8) = 1.0;
	}
	A.block<1, 8>(normalized.rows(), 0).setZero();
	A(normalized.rows(), 8) = 1;

	// Initialise solver.
	const JacobiSVD<Matrix<double, Dynamic, 9> >
		svd(A,  Eigen::ComputeFullU | Eigen::ComputeFullV);

	// Initiliase right hand side.
	Matrix<double, Dynamic, 1> rhs = Matrix<double, Dynamic, 1>::Zero(A.rows());
	rhs(rhs.rows() -1) = 1;

	// Solve via SVD
	const  MatrixXd parameters = svd.solve(rhs);

	// Write out result.
	Matrix<double, 3, 3> result;

	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			result(row, col) = parameters(3*row + col);
		}
	}

	// invert Scale

	std::cout << " --result --\n" << result << "\n ---\n";

	return input_scaling.transpose() * result * input_scaling;
};
