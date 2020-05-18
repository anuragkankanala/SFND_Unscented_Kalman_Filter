#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  //TODO change these to get a suitable NIS value
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  /**
   * DONE: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;

  x_ = VectorXd(n_x_);                 //state vector
  P_ = MatrixXd(n_x_, n_x_);           //state covariance matrix
  weights_ = VectorXd(2 * n_aug_ + 1); //weights matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  R_Radar_ = MatrixXd(3, 3);
  R_Laser_ = MatrixXd(2, 2);

  //Initializing P_ with a identity matrix
  P_ << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 1, 0,
      0, 0, 0, 0, 1;

  //Init weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (size_t i = 1; i < 2 * n_aug_ + 1; i++)
  {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  //Init Measurement Covarinace Matrices R
  R_Radar_.fill(0.0);
  R_Radar_(0, 0) = std_radr_ * std_radr_;
  R_Radar_(1, 1) = std_radphi_ * std_radphi_;
  R_Radar_(2, 2) = std_radrd_ * std_radrd_;

  R_Laser_.fill(0.0);
  R_Laser_(0, 0) = std_laspx_ * std_laspx_;
  R_Laser_(1, 1) = std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

void UKF::InitializeState(MeasurementPackage meas_package)
{
  //Process LIDAR
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
  {
    float px = meas_package.raw_measurements_[0];
    float py = meas_package.raw_measurements_[1];

    x_ << px,
        py,
        0,
        0,
        0;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  //Process Radar measuremnt
  else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
  {
    float rho = meas_package.raw_measurements_[0];
    float phi = meas_package.raw_measurements_[1];
    float rho_dot = meas_package.raw_measurements_[2];

    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    float px = rho * cos_phi;
    float py = rho * sin_phi;
    float v = sqrt(pow(rho_dot * cos_phi, 2) + pow(rho_dot * sin_phi, 2));

    x_ << px,
        py,
        v,
        0,
        0;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * DONE: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_)
  {
    InitializeState(meas_package);
    return;
  }

  float dT = (meas_package.timestamp_ - time_us_) / 1e6;
  Prediction(dT);

  //Process LIDAR
  if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
  {
    UpdateLidar(meas_package);
  }

  //Process Radar measuremnt
  else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
  {
    UpdateRadar(meas_package);
  }

  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}