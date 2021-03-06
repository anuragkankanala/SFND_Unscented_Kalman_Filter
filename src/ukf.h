#ifndef UKF_H
#define UKF_H

#include <iostream>
#include <fstream>
#include <vector>
#include "Eigen/Dense"
#include "measurement_package.h"

class UKF
{
public:
  /**
   * Constructor
   */
  UKF(bool logNIS = false);

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  /**
   * Initializes the state vector X_ for the first time with measurement.
   * @param meas_package First measurement received.
   */
  void InitializeState(MeasurementPackage meas_package);

  /**
   * Generates Sigma Points for the Prediction Step
   * @param sigma_points Reference to the matrix to populate the sigma points
   */
  void GenerateSigmaPoints(Eigen::MatrixXd &sigma_points);

  /**
   * Generates Augmented Sigma Points for the Prediction Step
   * @param sigma_points Reference to the matrix to populate the sigma points
   */
  void GenerateAugmentedSigmaPoints(Eigen::MatrixXd &sigma_points);

  /**
   * Predicts the sigma points by applying the process model to the 
   * generated sigma points.
   * @param sigma_points Reference to the generated sigma points
   * @param sigma_points_predicted Reference to the matrix to store the predicted points
   * @param delta_t time diff between measurements
   */
  void PredictSigmaPoints(const Eigen::MatrixXd &sigma_points, Eigen::MatrixXd &sigma_points_predicted, float delta_t);

  /**
   * Predicts the new mean state and covariance matrices from the predicted sigma points
   * @param sigma_points_predicted Reference to the predicted sigma points
   * @param x_new Referece to the matrix to store the new mean state
   * @param P_new Reference to the matrix to store the new covariance matrix
   */
  void PredictMeanAndCovariance(const Eigen::MatrixXd &sigma_points_predicted, Eigen::VectorXd &x_new, Eigen::MatrixXd &P_new);

  /**
   * Transforms the predicted state into the radar measurement space.
   * @param z_pred Reference to vector to store the mean predicted measurement
   * @param S Reference to the matrix to store measurement covariance matrix
   * @param ZSig Reference to matrix to hold sigma points in measurement space
   */
  void PredictRadarMeasurement(Eigen::VectorXd &z_pred, Eigen::MatrixXd &S, Eigen::MatrixXd &ZSig);

  /**
   * Transforms the predicted state into the lidar measurement space.
   * @param z_pred Reference to vector to store the mean predicted measurement
   * @param S Reference to the matrix to store measurement covariance matrix
   * @param ZSig Reference to matrix to hold sigma points in measurement space
   */
  void PredictLidarMeasurement(Eigen::VectorXd &z_pred, Eigen::MatrixXd &S, Eigen::MatrixXd &ZSig);

  /**
   * Calculates NIS for given values
   * @param Z_Meas Reference to the actual measurement by the sensor
   * @param Z_Pred Reference to the predicted measurement in sensor space
   * @param S Referece to the measurement covariance matrix
   */
  double CalculateNIS(Eigen::VectorXd &Z_Meas, Eigen::VectorXd &Z_Pred, Eigen::MatrixXd &S);

  /**
   * Writes Radar NIS values to csv file
   */
  void WriteRadarNISToCSV();

  /**
   * Writes Lidar NIS values to csv file
   */
  void WriteLidarNISToCSV();

  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Sigma point spreading parameter
  double lambda_;

  //Lidar measurement covariance matrix R
  Eigen::MatrixXd R_Laser_;

  //Radar measurement covariance matrix R
  Eigen::MatrixXd R_Radar_;

  //Log NIS
  bool logNIS_{false};
  double radar_95_NIS_limit = 7.815;
  double lidar_95_NIS_limit = 5.991;
  std::vector<double> radar_NIS;
  std::vector<double> lidar_NIS;
};

#endif // UKF_H