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
   * DONE: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // //Generate Sigma Points - without augmentation
  // MatrixXd sigma_points = MatrixXd(n_x_, 2 * n_x_ + 1);
  // GenerateSigmaPoints(sigma_points);

  //Generate Augmented Sigma Points
  MatrixXd X_Sig_Aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  GenerateAugmentedSigmaPoints(X_Sig_Aug);

  //Predict Sigma points by applying the process model
  PredictSigmaPoints(X_Sig_Aug, Xsig_pred_, delta_t);

  PredictMeanAndCovariance(Xsig_pred_, x_, P_);
}

void UKF::PredictMeanAndCovariance(const Eigen::MatrixXd &sigma_points_predicted, Eigen::VectorXd &x_new, Eigen::MatrixXd &P_new)
{
  //predict new mean state
  x_new.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    x_new = x_new + weights_(i) * sigma_points_predicted.col(i);
  }

  // predicted state covariance matrix
  P_new.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    // state difference
    VectorXd x_diff = sigma_points_predicted.col(i) - x_new;

    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_new = P_new + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::PredictSigmaPoints(const Eigen::MatrixXd &sigma_points, Eigen::MatrixXd &sigma_points_predicted, float delta_t)
{
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double px = sigma_points(0, i);
    double py = sigma_points(1, i);
    double v = sigma_points(2, i);
    double yaw = sigma_points(3, i);
    double yaw_dot = sigma_points(4, i);
    double nu_a = sigma_points(5, i);
    double nu_yaw_dot_dot = sigma_points(6, i);

    //predict state values
    double px_p, py_p;

    //check if yaw_dot is zero
    if (fabs(yaw_dot) > std::numeric_limits<double>::epsilon())
    {
      px_p = px + v / yaw_dot * (sin(yaw + yaw_dot * delta_t) - sin(yaw));
      py_p = py + v / yaw_dot * (cos(yaw) - cos(yaw + yaw_dot * delta_t));
    }

    else
    {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yaw_dot * delta_t;
    double yawd_p = yaw_dot;

    //add process noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yaw_dot_dot * delta_t * delta_t;
    yawd_p = yawd_p + nu_yaw_dot_dot * delta_t;

    //fill the output matrix
    sigma_points_predicted(0, i) = px_p;
    sigma_points_predicted(1, i) = py_p;
    sigma_points_predicted(2, i) = v_p;
    sigma_points_predicted(3, i) = yaw_p;
    sigma_points_predicted(4, i) = yawd_p;
  }
}

void UKF::GenerateAugmentedSigmaPoints(Eigen::MatrixXd &sigma_points)
{
  VectorXd x_aug = VectorXd(n_aug_);         //augmented mean vector
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_); //augmented state covairance matrix

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  sigma_points.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i)
  {
    sigma_points.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    sigma_points.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

void UKF::GenerateSigmaPoints(Eigen::MatrixXd &sigma_points)
{
  //calculate square root of P
  MatrixXd L = P_.llt().matrixL();

  //set first columns to state vector x_
  sigma_points.col(0) = x_;

  //remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    sigma_points.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * L.col(i);
    sigma_points.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * L.col(i);
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * DONE: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;
  VectorXd z_pred = VectorXd(n_z);               //mean predicted measurement
  MatrixXd S = MatrixXd(n_z, n_z);               //matrix for predicted measurement covariance
  VectorXd z_incoming = VectorXd(n_z);           //incoming new measurement from sensor
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // create matrix for sigma points in measurement space

  z_incoming << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1];

  PredictLidarMeasurement(z_pred, S, Zsig);

  MatrixXd Tc = MatrixXd(n_x_, n_z); // create matrix for cross correlation Tc

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {

    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z_incoming - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;

  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::PredictLidarMeasurement(Eigen::VectorXd &z_pred, Eigen::MatrixXd &S, Eigen::MatrixXd &ZSig)
{
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    // measurement model
    ZSig(0, i) = px;
    ZSig(1, i) = py;
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (size_t i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    z_pred = z_pred + weights_(i) * ZSig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (size_t i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd diff = (ZSig.col(i) - z_pred);
    S = S + weights_(i) * diff * diff.transpose();
  }
  S = S + R_Laser_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * DONE: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  VectorXd z_pred = VectorXd(n_z);               //mean predicted measurement
  MatrixXd S = MatrixXd(n_z, n_z);               //matrix for predicted measurement covariance
  VectorXd z_incoming = VectorXd(n_z);           //incoming new measurement from sensor
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1); // create matrix for sigma points in measurement space

  z_incoming << meas_package.raw_measurements_[0],
      meas_package.raw_measurements_[1],
      meas_package.raw_measurements_[2];

  PredictRadarMeasurement(z_pred, S, Zsig);

  MatrixXd Tc = MatrixXd(n_x_, n_z); // create matrix for cross correlation Tc

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {

    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z_incoming - z_pred;

  // angle normalization
  while (z_diff(1) > M_PI)
  {
    z_diff(1) -= 2. * M_PI;
  }

  while (z_diff(1) < -M_PI)
  {
    z_diff(1) += 2. * M_PI;
  }

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::PredictRadarMeasurement(Eigen::VectorXd &z_pred, Eigen::MatrixXd &S, Eigen::MatrixXd &ZSig)
{
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    ZSig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
    ZSig(1, i) = atan2(p_y, p_x);                                     // phi
    ZSig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    z_pred = z_pred + weights_(i) * ZSig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  { // 2n+1 simga points
    // residual
    VectorXd z_diff = ZSig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S = S + R_Radar_;
}