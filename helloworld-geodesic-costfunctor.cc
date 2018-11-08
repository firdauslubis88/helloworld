// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// Defaults should be suitable for a wide range of use cases, but
// better performance and accuracy might require tweaking.
static struct EstimateHomographyOptions {
	// Default settings for homography estimation which should be suitable
	// for a wide range of use cases.
	EstimateHomographyOptions()
		: max_num_iterations(50),
		expected_average_symmetric_distance(1e-16) {}

	// Maximal number of iterations for the refinement step.
	int max_num_iterations;

	// Expected average of symmetric geometric distance between
	// actual destination points and original ones transformed by
	// estimated homography matrix.
	//
	// Refinement will finish as soon as average of symmetric
	// geometric distance is less or equal to this value.
	//
	// This distance is measured in the same units as input points are.
	double expected_average_symmetric_distance;
};

static void degreeToDMS(float degrees, int& degree, int& minutes, int& seconds) {
	degree = static_cast<int>(degrees);
	minutes = static_cast<int>((degrees - degree) * 60);
	seconds = static_cast<int>((degrees - degree - minutes / 60) * 3600);
}

/**
 * Returns the distance between two points on the Earth.
 * Direct translation from http://www.movable-type.co.uk/scripts/latlong.html
 * @param lat1d Latitude of the first point in degrees
 * @param lon1d Longitude of the first point in degrees
 * @param lat2d Latitude of the second point in degrees
 * @param lon2d Longitude of the second point in degrees
 * @param radius radius of the sphere in distance units
 * @return The distance between the two points in distance units
 */
template <typename T>
static T distance(T lat1d, T lon1d, T lat2d, T lon2d) {
	T lat1r, lon1r, lat2r, lon2r, u, v;
	lat1r = lat1d;// lubis::ofDegToRad(lat1d);
	lon1r = lon1d;// lubis::ofDegToRad(lon1d);
	lat2r = lat2d;// lubis::ofDegToRad(lat2d);
	lon2r = lon2d;// lubis::ofDegToRad(lon2d);
	u = sin((lat2r - lat1r) * 0.5);
	v = sin((lon2r - lon1r) * 0.5);
	return 2.0 * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

/**
 * Returns the initial bearing between two points (from start point to end point) on the Earth.
 * Direct translation from http://www.movable-type.co.uk/scripts/latlong.html
 * @param lat1d Latitude of the start point in degrees
 * @param lon1d Longitude of the start point in degrees
 * @param lat2d Latitude of the end point in degrees
 * @param lon2d Longitude of the end point in degrees
 * @return The initial bearing from start point to end point in radians
 */
template <typename T>
static T bearing(T lat1d, T lon1d, T lat2d, T lon2d) {
	T lat1r, lon1r, lat2r, lon2r, deltaLat, deltaLong;
	lat1r = lat1d;// lubis::ofDegToRad(lat1d);
	lon1r = lon1d;// lubis::ofDegToRad(lon1d);
	lat2r = lat2d;// lubis::ofDegToRad(lat2d);
	lon2r = lon2d;// lubis::ofDegToRad(lon2d);
	deltaLat = lat2r - lat1r;
	deltaLong = lon2r - lon1r;
	return atan2(sin(deltaLong)*cos(lat2r), cos(lat1r)*sin(lat2r) - sin(lat1r)*cos(lat2r)*cos(deltaLong));
}

/**
 * Returns the distance between third point to a great circle formed by start point and end point (crosstrack-distance).
 * Direct translation from http://www.movable-type.co.uk/scripts/latlong.html
 * @param lat1d Latitude of the start point in degrees
 * @param lon1d Longitude of the start point in degrees
 * @param lat2d Latitude of the end point in degrees
 * @param lon2d Longitude of the end point in degrees
 * @param lat3d Latitude of the third point in degrees
 * @param lon3d Longitude of the third point in degrees
 * @param radius radius of the sphere in distance units
 * @return the distance between third point to a great circle formed by start point and end point in distance units.
 */
template <typename T>
static T crosstrackDistance(T lat1d, T lon1d, T lat2d, T lon2d, T lat3d, T lon3d) {
	T delta13, theta12, theta13;
	delta13 = distance(lat1d, lon1d, lat3d, lon3d);
	theta13 = bearing(lat1d, lon1d, lat3d, lon3d);
	theta12 = bearing(lat1d, lon1d, lat2d, lon2d);
	//		lubis::degreeToDMS(lubis::ofRadToDeg(theta13), deg, min, sec);
	//		lubis::degreeToDMS(lubis::ofRadToDeg(theta12), deg, min, sec);
	T a = sin(delta13), b = sin(theta13 - theta12);
	T c = asin(a*b);
	return c;
}

/**
 * Returns symmetric geodesic distance point a in coordinate A to its corresponding great circle in coordinate B after transformed by R and T matrix.
 * R and T matrix relate coordinate A with coordinate B. The geodesic distance is calculated twice (transformed point a with its great circle pair in B
 * and between transformed point b with its great circle in coordinate A.
 * Direct translation from http://www.movable-type.co.uk/scripts/latlong.html
 * @param INPUT R_t double pointer that consists of 6 arrays (3 for axis angle (to represent R) parameter and 3 for translation parameters).
 * @param INPUT x1 point a which distance with its corresponding great circle to be measured
 * @param INPUT x2 point a which distance with its corresponding great circle to be measured
 * @param OUTPUT forward_error the geodesic distance error between x1 with its great circle
 * @param OUTPUT backward_error the geodesic distance error between x2 with its great circle
 * @This is a templated function which can be used with double and J as their template
 */
template <typename T>
inline static void SymmetricGeodesicDistanceTerms(const T *R_t,
	const Eigen::Matrix<T, 3, 1> &x1,
	const Eigen::Matrix<T, 3, 1> &x2,
	T* forward_error,
	T* backward_error) {
	typedef Eigen::Matrix<T, 3, 1> Vec3;
	T x[3] = { x1(0), x1(1), x1(2) };
	T y[3] = { x2(0), x2(1), x2(2) };

	// Compute projective coordinates: transformed_x = RX + t.
	T transformed_x[3];

	ceres::AngleAxisRotatePoint(R_t, x, transformed_x);
	transformed_x[0] += R_t[3];
	transformed_x[1] += R_t[4];
	transformed_x[2] += R_t[5];

	T lat3 = atan(transformed_x[1] / sqrt(transformed_x[0] * transformed_x[0] + transformed_x[2] * transformed_x[2]));
	T lon3 = atan(transformed_x[0] / transformed_x[2]);
	T lat1 = atan(y[1] / sqrt(y[0] * y[0] + y[2] * y[2]));
	T lon1 = atan(y[0] / y[2]);
	T lat2 = atan(R_t[4] / sqrt(R_t[3] * R_t[3] + R_t[5] * R_t[5]));
	T lon2 = atan(R_t[3] / R_t[5]);
	*forward_error = crosstrackDistance<T>(lat1, lon1, lat2, lon2, lat3, lon3);

	// Compute projective coordinates: transformed_y = R(Y - t).
	T transformed_y[3];
	T tmp_y[3] = { y[0], y[1], y[2] };
	T inverted_R_t[3] = { -R_t[0], -R_t[1], -R_t[2] };
	tmp_y[0] -= R_t[3];
	tmp_y[1] -= R_t[4];
	tmp_y[2] -= R_t[5];
	ceres::AngleAxisRotatePoint(inverted_R_t, tmp_y, transformed_y);

	lat3 = atan(transformed_y[1] / sqrt(transformed_y[0] * transformed_y[0] + transformed_y[2] * transformed_y[2]));
	lon3 = atan(transformed_y[0] / transformed_y[2]);
	lat1 = atan(x[1] / sqrt(x[0] * x[0] + x[2] * x[2]));
	lon1 = atan(x[0] / x[2]);
	lat2 = atan(-R_t[4] / sqrt((-R_t[3])*(-R_t[3]) + (-R_t[5]) * (-R_t[5])));
	lon2 = atan(-R_t[3] / -R_t[5]);
	*backward_error = crosstrackDistance<T>(lat1, lon1, lat2, lon2, lat3, lon3);
}

/**
 * Returns symmetric geodesic distance point a in coordinate A to its corresponding great circle in coordinate B after transformed by R and T matrix.
 * R and T matrix relate coordinate A with coordinate B. The geodesic distance is calculated twice (transformed point a with its great circle pair in B
 * and between transformed point b with its great circle in coordinate A.
 * Direct translation from http://www.movable-type.co.uk/scripts/latlong.html
 * @param INPUT R_t double pointer that consists of 6 arrays (3 for axis angle (to represent R) parameter and 3 for translation parameters).
 * @param INPUT x1 point a which distance with its corresponding great circle to be measured
 * @param INPUT x2 point a which distance with its corresponding great circle to be measured
 * This return the symmetric geometric distance
 * Differ with SymmetricGeodesicDistanceTerms, this is specific for double type (not a template function)
 */
inline static double SymmetricGeodesicDistance(const double* R_t,
	const Eigen::Vector3d &x1,
	const Eigen::Vector3d &x2) {
	double forward_error, backward_error;
	SymmetricGeodesicDistanceTerms<double>(R_t,
		x1,
		x2,
		&forward_error,
		&backward_error);
	return forward_error +
		backward_error;
}

/*Cost functor class which computes symmetric geodesic distance used for estimating the R and T.
 *It overrides the '()' operator required to be called by ceres library.
 */
static class RTSymmetricGeodesicCostFunctor {
public:
	/*The constructor will initialize the x and y members variable.
	 *@param INPUT: x a 3D observarion points
	 *@param INPUT: y a 3D observarion points
	 *x and y are corresponding pair each other
	 */
	RTSymmetricGeodesicCostFunctor(const Eigen::Vector3d &x,
		const Eigen::Vector3d &y)
		: x_(x), y_(y) { }

	/*The templated '()' operator that is called by ceres.
	 *@param INPUT-OUTPUT: R_t double pointer that consists of 6 arrays (3 for axis angle (to represent R) parameter and 3 for translation parameters).
	 *@param INPUT-OUTPUT: residuals a pointer that contain the cost function result (symmetric geodesic distance result)
	 *This always return true.
	 */
	template<typename T>
	bool operator()(const T* R_t, T* residuals) const {
		typedef Eigen::Matrix<T, 3, 3> Mat3;
		typedef Eigen::Matrix<T, 3, 1> Vec3;

		Vec3 x(T(x_(0)), T(x_(1)), T(x_(2)));
		Vec3 y(T(y_(0)), T(y_(1)), T(y_(2)));

		SymmetricGeodesicDistanceTerms<T>(R_t,
			x,
			y,
			&residuals[0],
			&residuals[1]);
		return true;
	}

	const Eigen::Vector3d x_;
	const Eigen::Vector3d y_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  srand(time(0));
  int total_data = 100;
//  Eigen::MatrixXd x1(3, 100);
//  std::vector<std::vector<double>> x, transformed_x;
  Eigen::MatrixXd x(3, total_data), transformed_x(3, total_data);
  double R_t[6] = { 2 * (((double)rand() / (RAND_MAX)) - 0.5), 2 * (((double)rand() / (RAND_MAX)) - 0.5), 2 * (((double)rand() / (RAND_MAX)) - 0.5), 2 * (((double)rand() / (RAND_MAX)) - 0.5), 2 * (((double)rand() / (RAND_MAX)) - 0.5), 2 * (((double)rand() / (RAND_MAX)) - 0.5) },
//	  estimated_R_t[6] = { 0.0000001, 0, 0, 0.0000001, 0.0000001, 0.0000001 };
      estimated_R_t[6] = { R_t[0], R_t[1], R_t[2], R_t[3], R_t[4], R_t[5] };
  for (int i = 0; i < total_data; ++i) {
	  std::vector<double> tmpVec = { 2 * (((double)rand() / (RAND_MAX)) - 0.5) , 2 * (((double)rand() / (RAND_MAX)) - 0.5) , 2 * (((double)rand() / (RAND_MAX)) - 0.5) };
	  x(0, i) = 2 * (((double)rand() / (RAND_MAX)) - 0.5), x(1, i) = 2 * (((double)rand() / (RAND_MAX)) - 0.5), x(2, i) = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
	  double tmp_transformed_x[3];
	  ceres::AngleAxisRotatePoint(&R_t[0], x.col(i).data(), &tmp_transformed_x[0]);
	  tmp_transformed_x[0] += R_t[3];
	  tmp_transformed_x[1] += R_t[4];
	  tmp_transformed_x[2] += R_t[5];
	  // Apply some noise so algebraic estimation is not good enough.
	  tmp_transformed_x[0] += 0.01;//double(rand() % 1000) / 5000.0;
	  tmp_transformed_x[1] += 0.01;//double(rand() % 1000) / 5000.0;
	  tmp_transformed_x[2] += 0.01;//double(rand() % 1000) / 5000.0;
	  transformed_x(0, i) = { tmp_transformed_x[0] }, transformed_x(1, i) = { tmp_transformed_x[1] }, transformed_x(2, i) = { tmp_transformed_x[2] };
  }

//  for (int i = 0; i < total_data; ++i) {
//
//  }

//  for (size_t i = 0; i < 100; i++)
//  {
//	  std::cout << "x:\t" << x.col(i) << ", " << std::endl << "transformed_x:\t" << transformed_x.col(i) << std::endl;
//  }

  EstimateHomographyOptions options;
  options.expected_average_symmetric_distance = 0.0002;

  ceres::Problem problem;
  for (int i = 0; i < x.cols(); i++) {
	  RTSymmetricGeodesicCostFunctor
		  *rt_symmetric_geodesic_cost_function =
		  new RTSymmetricGeodesicCostFunctor(x.col(i),
			  transformed_x.col(i));

	  problem.AddResidualBlock(
		  new ceres::AutoDiffCostFunction<
		  RTSymmetricGeodesicCostFunctor,
		  2,  // num_residuals
		  6>(rt_symmetric_geodesic_cost_function),
		  NULL,
		  estimated_R_t);
  }

  // Configure the solve.
  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.trust_region_strategy_type = ceres::DOGLEG;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.update_state_every_iteration = true;

  // Terminate if the average symmetric distance is good enough.
//		AverageTerminationCheckingCallback callback(x1, x2, options, R, T);
//		solver_options.callbacks.push_back(&callback);
  solver_options.minimizer_progress_to_stdout = true;

	// Run the solve.
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  //		LOG(INFO) << "Summary:\n" << summary.FullReport();
  std::cout << "True R_t:\n"      << R_t[0] << ", " << R_t[1] << ", " << R_t[2] << ", " << R_t[3] << ", " << R_t[4] << ", " << R_t[5] << std::endl;
  std::cout << "Estimated R_t:\n" << estimated_R_t[0] << ", " << estimated_R_t[1] << ", " << estimated_R_t[2] << ", " << estimated_R_t[3] << ", " << estimated_R_t[4] << ", " << estimated_R_t[5] << std::endl;

  return summary.IsSolutionUsable();

  return 0;
}
