-- phpMyAdmin SQL Dump
-- version 4.9.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Sep 20, 2023 at 07:09 AM
-- Server version: 10.4.10-MariaDB
-- PHP Version: 7.3.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `lungss_cancer_detection`
--

-- --------------------------------------------------------

--
-- Table structure for table `auth_group`
--

DROP TABLE IF EXISTS `auth_group`;
CREATE TABLE IF NOT EXISTS `auth_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_group_permissions`
--

DROP TABLE IF EXISTS `auth_group_permissions`;
CREATE TABLE IF NOT EXISTS `auth_group_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissions_group_id_b120cbf9` (`group_id`),
  KEY `auth_group_permissions_permission_id_84c5c92e` (`permission_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_permission`
--

DROP TABLE IF EXISTS `auth_permission`;
CREATE TABLE IF NOT EXISTS `auth_permission` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  KEY `auth_permission_content_type_id_2f476e4b` (`content_type_id`)
) ENGINE=MyISAM AUTO_INCREMENT=45 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `auth_permission`
--

INSERT INTO `auth_permission` (`id`, `name`, `content_type_id`, `codename`) VALUES
(1, 'Can add log entry', 1, 'add_logentry'),
(2, 'Can change log entry', 1, 'change_logentry'),
(3, 'Can delete log entry', 1, 'delete_logentry'),
(4, 'Can view log entry', 1, 'view_logentry'),
(5, 'Can add permission', 2, 'add_permission'),
(6, 'Can change permission', 2, 'change_permission'),
(7, 'Can delete permission', 2, 'delete_permission'),
(8, 'Can view permission', 2, 'view_permission'),
(9, 'Can add group', 3, 'add_group'),
(10, 'Can change group', 3, 'change_group'),
(11, 'Can delete group', 3, 'delete_group'),
(12, 'Can view group', 3, 'view_group'),
(13, 'Can add user', 4, 'add_user'),
(14, 'Can change user', 4, 'change_user'),
(15, 'Can delete user', 4, 'delete_user'),
(16, 'Can view user', 4, 'view_user'),
(17, 'Can add content type', 5, 'add_contenttype'),
(18, 'Can change content type', 5, 'change_contenttype'),
(19, 'Can delete content type', 5, 'delete_contenttype'),
(20, 'Can view content type', 5, 'view_contenttype'),
(21, 'Can add session', 6, 'add_session'),
(22, 'Can change session', 6, 'change_session'),
(23, 'Can delete session', 6, 'delete_session'),
(24, 'Can view session', 6, 'view_session'),
(25, 'Can add last_login', 7, 'add_last_login'),
(26, 'Can change last_login', 7, 'change_last_login'),
(27, 'Can delete last_login', 7, 'delete_last_login'),
(28, 'Can view last_login', 7, 'view_last_login'),
(29, 'Can add user', 8, 'add_user'),
(30, 'Can change user', 8, 'change_user'),
(31, 'Can delete user', 8, 'delete_user'),
(32, 'Can view user', 8, 'view_user'),
(33, 'Can add manage_users_model', 9, 'add_manage_users_model'),
(34, 'Can change manage_users_model', 9, 'change_manage_users_model'),
(35, 'Can delete manage_users_model', 9, 'delete_manage_users_model'),
(36, 'Can view manage_users_model', 9, 'view_manage_users_model'),
(37, 'Can add dataset', 10, 'add_dataset'),
(38, 'Can change dataset', 10, 'change_dataset'),
(39, 'Can delete dataset', 10, 'delete_dataset'),
(40, 'Can view dataset', 10, 'view_dataset'),
(41, 'Can add feedback', 11, 'add_feedback'),
(42, 'Can change feedback', 11, 'change_feedback'),
(43, 'Can delete feedback', 11, 'delete_feedback'),
(44, 'Can view feedback', 11, 'view_feedback');

-- --------------------------------------------------------

--
-- Table structure for table `auth_user`
--

DROP TABLE IF EXISTS `auth_user`;
CREATE TABLE IF NOT EXISTS `auth_user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(150) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_groups`
--

DROP TABLE IF EXISTS `auth_user_groups`;
CREATE TABLE IF NOT EXISTS `auth_user_groups` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_groups_user_id_group_id_94350c0c_uniq` (`user_id`,`group_id`),
  KEY `auth_user_groups_user_id_6a12ed8b` (`user_id`),
  KEY `auth_user_groups_group_id_97559544` (`group_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_user_permissions`
--

DROP TABLE IF EXISTS `auth_user_user_permissions`;
CREATE TABLE IF NOT EXISTS `auth_user_user_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` (`user_id`,`permission_id`),
  KEY `auth_user_user_permissions_user_id_a95ead1b` (`user_id`),
  KEY `auth_user_user_permissions_permission_id_1fbb5f2c` (`permission_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `django_admin_log`
--

DROP TABLE IF EXISTS `django_admin_log`;
CREATE TABLE IF NOT EXISTS `django_admin_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext DEFAULT NULL,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) UNSIGNED NOT NULL CHECK (`action_flag` >= 0),
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6` (`user_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `django_content_type`
--

DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE IF NOT EXISTS `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=MyISAM AUTO_INCREMENT=12 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_content_type`
--

INSERT INTO `django_content_type` (`id`, `app_label`, `model`) VALUES
(1, 'admin', 'logentry'),
(2, 'auth', 'permission'),
(3, 'auth', 'group'),
(4, 'auth', 'user'),
(5, 'contenttypes', 'contenttype'),
(6, 'sessions', 'session'),
(7, 'mainapp', 'last_login'),
(8, 'mainapp', 'user'),
(9, 'adminapp', 'manage_users_model'),
(10, 'userapp', 'dataset'),
(11, 'userapp', 'feedback');

-- --------------------------------------------------------

--
-- Table structure for table `django_migrations`
--

DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE IF NOT EXISTS `django_migrations` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=22 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_migrations`
--

INSERT INTO `django_migrations` (`id`, `app`, `name`, `applied`) VALUES
(1, 'contenttypes', '0001_initial', '2023-09-16 10:11:09.609525'),
(2, 'auth', '0001_initial', '2023-09-16 10:11:10.429005'),
(3, 'admin', '0001_initial', '2023-09-16 10:11:10.651825'),
(4, 'admin', '0002_logentry_remove_auto_add', '2023-09-16 10:11:10.667443'),
(5, 'admin', '0003_logentry_add_action_flag_choices', '2023-09-16 10:11:10.688147'),
(6, 'adminapp', '0001_initial', '2023-09-16 10:11:10.720551'),
(7, 'contenttypes', '0002_remove_content_type_name', '2023-09-16 10:11:10.858967'),
(8, 'auth', '0002_alter_permission_name_max_length', '2023-09-16 10:11:10.920014'),
(9, 'auth', '0003_alter_user_email_max_length', '2023-09-16 10:11:10.990667'),
(10, 'auth', '0004_alter_user_username_opts', '2023-09-16 10:11:11.018248'),
(11, 'auth', '0005_alter_user_last_login_null', '2023-09-16 10:11:11.087381'),
(12, 'auth', '0006_require_contenttypes_0002', '2023-09-16 10:11:11.100877'),
(13, 'auth', '0007_alter_validators_add_error_messages', '2023-09-16 10:11:11.126129'),
(14, 'auth', '0008_alter_user_username_max_length', '2023-09-16 10:11:11.207043'),
(15, 'auth', '0009_alter_user_last_name_max_length', '2023-09-16 10:11:11.286634'),
(16, 'auth', '0010_alter_group_name_max_length', '2023-09-16 10:11:11.351887'),
(17, 'auth', '0011_update_proxy_permissions', '2023-09-16 10:11:11.370827'),
(18, 'auth', '0012_alter_user_first_name_max_length', '2023-09-16 10:11:11.459914'),
(19, 'mainapp', '0001_initial', '2023-09-16 10:11:11.522508'),
(20, 'sessions', '0001_initial', '2023-09-16 10:11:11.603074'),
(21, 'userapp', '0001_initial', '2023-09-16 10:11:11.771641');

-- --------------------------------------------------------

--
-- Table structure for table `django_session`
--

DROP TABLE IF EXISTS `django_session`;
CREATE TABLE IF NOT EXISTS `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_session`
--

INSERT INTO `django_session` (`session_key`, `session_data`, `expire_date`) VALUES
('is6g4nizoz8a9bzoh5nduhd3p7kf4zd4', 'eyJFbWFpbCI6InVzZXJAMTIzIiwiVXNlcl9pZCI6M30:1qir7K:INeyMdwo_V9wmIpoOPiQGiHmroxh3k9Y9v7JYfP-LXc', '2023-10-04 06:55:22.881710');

-- --------------------------------------------------------

--
-- Table structure for table `feedback_details`
--

DROP TABLE IF EXISTS `feedback_details`;
CREATE TABLE IF NOT EXISTS `feedback_details` (
  `Feed_id` int(11) NOT NULL AUTO_INCREMENT,
  `Rating` varchar(100) DEFAULT NULL,
  `Review` varchar(225) DEFAULT NULL,
  `Sentiment` varchar(100) DEFAULT NULL,
  `datetime` datetime(6) NOT NULL,
  `Reviewer_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`Feed_id`),
  KEY `feedback_details_Reviewer_id_13cf49be` (`Reviewer_id`)
) ENGINE=MyISAM AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `feedback_details`
--

INSERT INTO `feedback_details` (`Feed_id`, `Rating`, `Review`, `Sentiment`, `datetime`, `Reviewer_id`) VALUES
(1, '3', 'Good', 'positive', '2023-09-16 10:15:31.242465', 1),
(2, '2', 'I am not happy this app', 'negative', '2023-09-20 06:56:45.399820', 3);

-- --------------------------------------------------------

--
-- Table structure for table `last_login`
--

DROP TABLE IF EXISTS `last_login`;
CREATE TABLE IF NOT EXISTS `last_login` (
  `Id` int(11) NOT NULL AUTO_INCREMENT,
  `Login_Time` datetime(6) DEFAULT NULL,
  PRIMARY KEY (`Id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `manage_users`
--

DROP TABLE IF EXISTS `manage_users`;
CREATE TABLE IF NOT EXISTS `manage_users` (
  `User_id` int(11) NOT NULL AUTO_INCREMENT,
  `user_Profile` varchar(100) NOT NULL,
  `User_Email` varchar(50) NOT NULL,
  `User_Status` varchar(10) NOT NULL,
  PRIMARY KEY (`User_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `upload`
--

DROP TABLE IF EXISTS `upload`;
CREATE TABLE IF NOT EXISTS `upload` (
  `Data_id` int(11) NOT NULL AUTO_INCREMENT,
  `Image` varchar(100) NOT NULL,
  PRIMARY KEY (`Data_id`)
) ENGINE=MyISAM AUTO_INCREMENT=46 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `upload`
--

INSERT INTO `upload` (`Data_id`, `Image`) VALUES
(36, 'media/n2_XazP8fH.jpeg'),
(35, 'media/n2_syW4OZb.jpeg'),
(34, 'media/n3_CuJPC8S.jpeg'),
(33, 'media/person1947_bacteria_4876_fkslQXh.jpeg'),
(32, 'media/n2_6GbrNnc.jpeg'),
(31, 'media/person1950_bacteria_4881_AMaSO9Q.jpeg'),
(30, 'media/person1947_bacteria_4876_0xQ13ao.jpeg'),
(29, 'media/n2_8TloEnd.jpeg'),
(28, 'media/n2_JDy7k8t.jpeg'),
(27, 'media/person1946_bacteria_4875.jpeg'),
(26, 'media/n7_YNubMvr.jpeg'),
(37, 'media/person1947_bacteria_4876_39osz7V.jpeg'),
(38, 'media/person1947_bacteria_4876_poYeyWM.jpeg'),
(39, 'media/person1947_bacteria_4876_vSrzqDR.jpeg'),
(40, 'media/person1947_bacteria_4876_LuutGX2.jpeg'),
(41, 'media/n3_0SS1Tul.jpeg'),
(42, 'media/n4_ImSdRTf.jpeg'),
(43, 'media/n8_jqPwQzu.jpeg'),
(44, 'media/person1947_bacteria_4876_PhjEMON.jpeg'),
(45, 'media/person1947_bacteria_4876_Jcms2rJ.jpeg');

-- --------------------------------------------------------

--
-- Table structure for table `user`
--

DROP TABLE IF EXISTS `user`;
CREATE TABLE IF NOT EXISTS `user` (
  `User_id` int(11) NOT NULL AUTO_INCREMENT,
  `Full_name` longtext NOT NULL,
  `Image` varchar(100) DEFAULT NULL,
  `Email` varchar(50) DEFAULT NULL,
  `Address` longtext DEFAULT NULL,
  `Age` int(11) DEFAULT NULL,
  `Phone_Number` longtext DEFAULT NULL,
  `Password` longtext DEFAULT NULL,
  `Date_Time` datetime(6) DEFAULT NULL,
  `User_Status` longtext DEFAULT NULL,
  `Otp_Num` int(11) DEFAULT NULL,
  `Otp_Status` longtext DEFAULT NULL,
  `Last_Login_Time` time(6) DEFAULT NULL,
  `Last_Login_Date` date DEFAULT NULL,
  `No_Of_Times_Login` int(11) DEFAULT NULL,
  `Message` longtext DEFAULT NULL,
  PRIMARY KEY (`User_id`)
) ENGINE=MyISAM AUTO_INCREMENT=4 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `user`
--

INSERT INTO `user` (`User_id`, `Full_name`, `Image`, `Email`, `Address`, `Age`, `Phone_Number`, `Password`, `Date_Time`, `User_Status`, `Otp_Num`, `Otp_Status`, `Last_Login_Time`, `Last_Login_Date`, `No_Of_Times_Login`, `Message`) VALUES
(1, 'prashanthi', 'images/n6.jpeg', 'pprashanthi169@gmail.com', '8-3-232/b/50/c/6 , venkatagiri, yousufguda , hyderabad', 21, '9949803766', 'Amma@123', '2023-09-20 06:31:33.219426', 'accepted', 4683, 'verified', '12:01:33.000000', '2023-09-20', 14, NULL),
(3, 'user1', 'images/rose_Xw57vkX.jpg', 'user@123', '8-3-232/b/50/c/6 , venkatagiri, yousufguda , hyderabad', 21, '9949803766', 'Anusha@123', '2023-09-20 06:56:50.854883', 'accepted', 4112, 'verified', '12:26:50.000000', '2023-09-20', 1, NULL);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
