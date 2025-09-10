#include "so_arm.h"
#define Env SoArm
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    // Required numeric params
    env->dof = (int)unpack(kwargs, "dof");
    if (env->dof <= 0 || env->dof > SOARM_MAX_DOF) {
        PyErr_SetString(PyExc_ValueError, "Invalid dof");
        return 1;
    }
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->grasp_radius = (float)unpack(kwargs, "grasp_radius");
    // Optional: object radius; default small ball
    PyObject* py_obj_r = PyDict_GetItemString(kwargs, "obj_radius");
    env->obj_radius = py_obj_r ? (float)PyFloat_AsDouble(py_obj_r) : 0.02f;

    // DH arrays and limits (passed as Python lists/tuples)
    PyObject* py_a = PyDict_GetItemString(kwargs, "dh_a");
    PyObject* py_alpha = PyDict_GetItemString(kwargs, "dh_alpha");
    PyObject* py_d = PyDict_GetItemString(kwargs, "dh_d");
    PyObject* py_jmin = PyDict_GetItemString(kwargs, "joint_min");
    PyObject* py_jmax = PyDict_GetItemString(kwargs, "joint_max");
    PyObject* py_wsmin = PyDict_GetItemString(kwargs, "ws_min");
    PyObject* py_wsmax = PyDict_GetItemString(kwargs, "ws_max");
    PyObject* py_tau   = PyDict_GetItemString(kwargs, "servo_tau");
    PyObject* py_rate  = PyDict_GetItemString(kwargs, "joint_rate");
    PyObject* py_jaw_min = PyDict_GetItemString(kwargs, "jaw_min");
    PyObject* py_jaw_max = PyDict_GetItemString(kwargs, "jaw_max");
    PyObject* py_grip_speed = PyDict_GetItemString(kwargs, "grip_speed");
    PyObject* py_ee_radius = PyDict_GetItemString(kwargs, "ee_radius");

    if (!PySequence_Check(py_a) || !PySequence_Check(py_alpha) || !PySequence_Check(py_d)
        || !PySequence_Check(py_jmin) || !PySequence_Check(py_jmax)
        || !PySequence_Check(py_wsmin) || !PySequence_Check(py_wsmax)) {
        PyErr_SetString(PyExc_TypeError, "DH/limit/workspace params must be sequences");
        return 1;
    }

    for (int i=0;i<env->dof;i++){
        env->dh_a[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_a, i));
        env->dh_alpha[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_alpha, i));
        env->dh_d[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_d, i));
        env->joint_min[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_jmin, i));
        env->joint_max[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_jmax, i));
    }
    for (int k=0;k<3;k++){
        env->ws_min[k] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_wsmin, k));
        env->ws_max[k] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_wsmax, k));
    }

    // Optional servo model parameters
    if (py_tau && PySequence_Check(py_tau)){
        for (int i=0;i<env->dof;i++){
            env->servo_tau[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_tau, i));
        }
    } else {
        for (int i=0;i<env->dof;i++) env->servo_tau[i] = 0.0f;
    }
    if (py_rate && PySequence_Check(py_rate)){
        for (int i=0;i<env->dof;i++){
            env->joint_rate[i] = (float)PyFloat_AsDouble(PySequence_Fast_GET_ITEM(py_rate, i));
        }
    } else {
        for (int i=0;i<env->dof;i++) env->joint_rate[i] = 0.05f;
    }
    // Optional gripper parameters
    env->jaw_min = py_jaw_min ? (float)PyFloat_AsDouble(py_jaw_min) : 0.0f;
    env->jaw_max = py_jaw_max ? (float)PyFloat_AsDouble(py_jaw_max) : 0.04f;
    env->grip_speed = py_grip_speed ? (float)PyFloat_AsDouble(py_grip_speed) : 0.004f;
    env->ee_radius = py_ee_radius ? (float)PyFloat_AsDouble(py_ee_radius) : 0.0f;

    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "success_rate", log->success_rate);
    assign_to_dict(dict, "dist_to_goal", log->dist_to_goal);
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "total_grasps", log->total_grasps);
    assign_to_dict(dict, "total_places", log->total_places);
    return 0;
}


