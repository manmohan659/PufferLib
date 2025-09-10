#pragma once

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#define SOARM_MAX_DOF 8

typedef struct Log Log;
struct Log {
    float episode_return;
    float episode_length;
    float success_rate;
    float dist_to_goal;
    float n;
    float total_grasps;
    float total_places;
};

typedef struct SoArm SoArm;
struct SoArm {
    // Required pointers (populated by env_binding)
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;

    // Logging
    Log log;

    // Config
    int dof;                // number of joints (<= SOARM_MAX_DOF)
    int max_steps;          // steps before timeout
    float grasp_radius;     // grasp distance threshold (m)
    float obj_radius;       // radius of fetch ball (m)
    float joint_min[SOARM_MAX_DOF];
    float joint_max[SOARM_MAX_DOF];
    // Optional simple servo model: per-joint first-order lag and rate limit (rad/step)
    float servo_tau[SOARM_MAX_DOF];    // time constant in steps; 0 => no lag
    float joint_rate[SOARM_MAX_DOF];   // max delta per step after lag (rad/step)
    float dh_a[SOARM_MAX_DOF];
    float dh_alpha[SOARM_MAX_DOF];
    float dh_d[SOARM_MAX_DOF];
    float ws_min[3];
    float ws_max[3];
    float ee_radius;       // visual radius for EE marker; 0 to disable

    // Parallel-jaw gripper state and parameters (world units: meters)
    float jaw_width;     // current opening (m)
    float jaw_min;       // fully closed width (m)
    float jaw_max;       // fully open width (m)
    float grip_speed;    // opening/closing speed per step (m/step), acted by actions[dof]

    // State
    float q[SOARM_MAX_DOF]; // joint angles (rad)
    float ee[3];            // end-effector position
    float obj[3];           // object position
    float goal[3];          // goal position
    unsigned char grasped;  // 0/1 flag
    int t;                  // step counter
    // Optional renderer state
    void* client;

    // Reward shaping state/params
    float prev_ee_to_obj;   // previous distance EE->object
    float prev_obj_to_goal; // previous distance object->goal
    float w_approach_obj;   // small shaping weight
    float w_approach_goal;  // shaping when grasped
    float bonus_grasp;      // event bonus
    float bonus_place;      // success bonus (object in box)
    float bonus_idle;       // idle success bonus if already solved and no movement
    float action_penalty;   // sqrt(l2) multiplier

    // Goal box dimensions (AABB centered at goal)
    float box_half;         // half-size in X and Y (square footprint)
    float box_height;       // wall height (for render and place condition)

    // Cumulative counters and last event state
    unsigned char was_grasped;
    int total_grasps;
    int total_places;
};

// Helpers and core lifecycle (implemented inline to satisfy single-file build)
static inline void compute_link_positions(const SoArm* env, const float* q, float* out_xyz);

static inline float clampf(float x, float lo, float hi){
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline float randf(float lo, float hi){
    float r = (float)rand()/(float)RAND_MAX;
    return lo + r*(hi-lo);
}

static inline void mat4_mul(const float* A, const float* B, float* C){
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            float s = 0.0f;
            for(int k=0;k<4;k++) s += A[i*4+k]*B[k*4+j];
            C[i*4+j] = s;
        }
    }
}

static inline void dh_A(float a, float alpha, float d, float theta, float* T){
    float ca = cosf(alpha), sa = sinf(alpha);
    float ct = cosf(theta), st = sinf(theta);
    T[0] = ct;     T[1] = -st*ca; T[2] =  st*sa; T[3] = a*ct;
    T[4] = st;     T[5] =  ct*ca; T[6] = -ct*sa; T[7] = a*st;
    T[8] = 0.0f;   T[9] =  sa;    T[10]=  ca;    T[11]= d;
    T[12]= 0.0f;   T[13]= 0.0f;   T[14]= 0.0f;   T[15]= 1.0f;
}

// Distance from point p to segment ab; optionally outputs closest point cp (size 3)
static inline float dist_point_seg3(const float* p, const float* a, const float* b, float* cp){
    float vx = b[0]-a[0], vy=b[1]-a[1], vz=b[2]-a[2];
    float wx = p[0]-a[0], wy=p[1]-a[1], wz=p[2]-a[2];
    float vv = vx*vx+vy*vy+vz*vz;
    float t = vv > 1e-8f ? (vx*wx+vy*wy+vz*wz)/vv : 0.0f;
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    float cx = a[0] + t*vx, cy = a[1] + t*vy, cz = a[2] + t*vz;
    if (cp){ cp[0]=cx; cp[1]=cy; cp[2]=cz; }
    float dx = p[0]-cx, dy=p[1]-cy, dz=p[2]-cz;
    return sqrtf(dx*dx+dy*dy+dz*dz);
}

static inline void fk_pos(const SoArm* env, const float* q, float* ee_out3){
    float T[16] = {1,0,0,0,
                   0,1,0,0,
                   0,0,1,0,
                   0,0,0,1};
    float A[16], C[16];
    for(int i=0;i<env->dof;i++){
        dh_A(env->dh_a[i], env->dh_alpha[i], env->dh_d[i], q[i], A);
        mat4_mul(T, A, C);
        memcpy(T, C, sizeof(float)*16);
    }
    ee_out3[0] = T[3];
    ee_out3[1] = T[7];
    ee_out3[2] = T[11];
}

// Return minimum z across link endpoints and a few midpoints for quick plane checks
static inline float min_z_links(const SoArm* env, const float* pts){
    float minz = 1e9f;
    for (int i=0;i<env->dof;i++){
        const float* a = &pts[i*3];
        const float* b = &pts[(i+1)*3];
        // endpoints
        if (a[2] < minz) minz = a[2];
        if (b[2] < minz) minz = b[2];
        // a few midpoints along the segment
        for (int s=1;s<=3;s++){
            float t = (float)s/4.0f;
            float z = a[2] + t*(b[2]-a[2]);
            if (z < minz) minz = z;
        }
    }
    return minz;
}

static inline void write_obs(const SoArm* env){
    int n = env->dof;
    float* o = env->observations;
    for(int i=0;i<n;i++) o[i] = env->q[i];
    o[n+0] = env->ee[0]; o[n+1] = env->ee[1]; o[n+2] = env->ee[2];
    o[n+3] = env->obj[0]; o[n+4] = env->obj[1]; o[n+5] = env->obj[2];
    o[n+6] = env->goal[0]; o[n+7] = env->goal[1]; o[n+8] = env->goal[2];
    o[n+9] = env->grasped ? 1.0f : 0.0f;
}

static inline void init(SoArm* env){
    // Default reward weights
    env->w_approach_obj = 0.1f;
    env->w_approach_goal = 0.2f;
    env->bonus_grasp = 1.0f;
    env->bonus_place = 2.0f;
    env->bonus_idle = 1.0f; // half credit; base place is 2.0
    env->action_penalty = 0.001f;
    // Default goal box dims (0.14 x 0.14 footprint, 3cm tall walls)
    env->box_half = 0.07f;
    env->box_height = 0.03f;
    env->was_grasped = 0;
    env->total_grasps = 0;
    env->total_places = 0;
}

static inline void c_reset(SoArm* env){
    for(int i=0;i<env->dof;i++){
        float span = 0.2f;
        env->q[i] = clampf(randf(-span, span), env->joint_min[i], env->joint_max[i]);
    }
    fk_pos(env, env->q, env->ee);
    // sample object/goal with margins and keep-out near links
    for(int k=0;k<3;k++){
        env->goal[k] = randf(env->ws_min[k], env->ws_max[k]);
    }
    if (env->goal[2] < env->obj_radius) env->goal[2] = env->obj_radius;

    // compute link positions for keep-out checks
    float pts[(SOARM_MAX_DOF+1)*3];
    compute_link_positions(env, env->q, pts);
    float keepout = env->obj_radius + 0.012f; // link visual radius ~1.2cm
    int attempts = 0;
    while (1) {
        env->obj[0] = randf(env->ws_min[0], env->ws_max[0]);
        env->obj[1] = randf(env->ws_min[1], env->ws_max[1]);
        env->obj[2] = randf(env->ws_min[2], env->ws_max[2]);
        if (env->obj[2] < env->obj_radius) env->obj[2] = env->obj_radius;
        // check clearance from all link segments
        float p[3] = {env->obj[0], env->obj[1], env->obj[2]};
        float min_d = 1e9f;
        for (int i=0;i<env->dof;i++){
            const float* a = &pts[i*3];
            const float* b = &pts[(i+1)*3];
            float d = dist_point_seg3(p, a, b, NULL);
            if (d < min_d) min_d = d;
        }
        if (min_d >= keepout) break;
        attempts++;
        if (attempts > 64) {
            // fallback: place near EE offset
            env->obj[0] = env->ee[0];
            env->obj[1] = env->ee[1] + keepout;
            env->obj[2] = env->ee[2];
            break;
        }
    }
    env->grasped = 0;
    env->jaw_min = (env->jaw_min > 0.0f) ? env->jaw_min : 0.0f;
    env->jaw_max = (env->jaw_max > 0.0f) ? env->jaw_max : 0.04f;
    env->grip_speed = (env->grip_speed > 0.0f) ? env->grip_speed : 0.004f;
    env->jaw_width = env->jaw_max;
    // Initialize shaping distances
    float dxog = env->obj[0]-env->goal[0];
    float dyog = env->obj[1]-env->goal[1];
    float dzog = env->obj[2]-env->goal[2];
    env->prev_obj_to_goal = sqrtf(dxog*dxog + dyog*dyog + dzog*dzog);
    float dxeo = env->ee[0]-env->obj[0];
    float dyeo = env->ee[1]-env->obj[1];
    float dzeo = env->ee[2]-env->obj[2];
    env->prev_ee_to_obj = sqrtf(dxeo*dxeo + dyeo*dyeo + dzeo*dzeo);
    env->t = 0;
    write_obs(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
}

static inline void c_step(SoArm* env){
    int n = env->dof;
    // Desired joint change from action space
    float desired[SOARM_MAX_DOF];
    for (int i=0;i<n;i++){
        float dq_cmd = env->actions[i];
        if (!isfinite(dq_cmd)) dq_cmd = 0.0f;
        // Default rate if not provided
        float rate = env->joint_rate[i] != 0.0f ? env->joint_rate[i] : 0.05f;
        dq_cmd = clampf(dq_cmd, -rate, rate);
        desired[i] = dq_cmd;
    }

    // Apply simple first-order lag and joint limits, with floor constraint via backtracking
    float q_prev[SOARM_MAX_DOF];
    for (int i=0;i<n;i++) q_prev[i] = env->q[i];

    // initial target without floor constraint
    float alpha_i[SOARM_MAX_DOF];
    for (int i=0;i<n;i++){
        float tau = env->servo_tau[i];
        float alpha = (tau <= 0.0f) ? 1.0f : (1.0f/(tau + 1.0f)); // dt=1 step
        alpha_i[i] = alpha;
    }

    // compute proposal
    float q_prop[SOARM_MAX_DOF];
    for (int i=0;i<n;i++){
        float q_target = clampf(env->q[i] + desired[i], env->joint_min[i], env->joint_max[i]);
        q_prop[i] = env->q[i] + alpha_i[i] * (q_target - env->q[i]);
    }

    // Backtracking to keep links above ground plane z>=0
    float scale = 1.0f;
    for (int iter=0; iter<8; iter++){
        float q_try[SOARM_MAX_DOF];
        for (int i=0;i<n;i++) q_try[i] = q_prev[i] + scale*(q_prop[i]-q_prev[i]);
        float pts[(SOARM_MAX_DOF+1)*3];
        compute_link_positions(env, q_try, pts);
        if (min_z_links(env, pts) >= 0.0f){
            // accept
            for (int i=0;i<n;i++) env->q[i] = q_try[i];
            break;
        }
        scale *= 0.5f;
        if (iter == 7){
            // If still colliding with floor, keep previous q
            for (int i=0;i<n;i++) env->q[i] = q_prev[i];
        }
    }

    // Gripper command: open/close velocity control
    float grip = env->actions[n];
    if (!isfinite(grip)) grip = 0.0f;
    env->jaw_width = clampf(env->jaw_width + env->grip_speed * grip, env->jaw_min, env->jaw_max);
    fk_pos(env, env->q, env->ee);

    // Simple parallel-jaw grasp logic
    float dx = env->ee[0]-env->obj[0];
    float dy = env->ee[1]-env->obj[1];
    float dz = env->ee[2]-env->obj[2];
    float d = sqrtf(dx*dx + dy*dy + dz*dz);
    float effective_gap = env->jaw_width - 2.0f*env->obj_radius;
    if (!env->grasped){
        // Require being near the EE and jaws sufficiently closed to hold the sphere
        if (d <= env->grasp_radius + env->obj_radius && effective_gap <= 0.004f){
            env->grasped = 1;
        }
    } else {
        // Release if jaws open beyond sphere diameter
        if (env->jaw_width > 2.0f*env->obj_radius + 0.002f){
            env->grasped = 0;
        }
    }
    if (env->grasped){
        env->obj[0] = env->ee[0];
        env->obj[1] = env->ee[1];
        env->obj[2] = env->ee[2];
    } else {
        // prevent object ending inside links: push out of nearest segment if overlapping
        float pts[(SOARM_MAX_DOF+1)*3];
        compute_link_positions(env, env->q, pts);
        float p[3] = {env->obj[0], env->obj[1], env->obj[2]};
        float keepout = env->obj_radius + 0.012f;
        float cp[3];
        float min_d = 1e9f; int min_i = -1;
        for (int i=0;i<env->dof;i++){
            const float* a = &pts[i*3];
            const float* b = &pts[(i+1)*3];
            float d = dist_point_seg3(p, a, b, cp);
            if (d < min_d){ min_d = d; min_i = i; }
        }
        if (min_d < keepout && min_i >= 0){
            // push outward along normal
            const float* a = &pts[min_i*3];
            const float* b = &pts[(min_i+1)*3];
            (void)dist_point_seg3(p, a, b, cp);
            float nx = p[0]-cp[0], ny = p[1]-cp[1], nz = p[2]-cp[2];
            float nn = sqrtf(nx*nx+ny*ny+nz*nz);
            if (nn < 1e-6f){ nx = 0.0f; ny = 0.0f; nz = 1.0f; nn = 1.0f; }
            float scale = (keepout - min_d) + 1e-3f;
            env->obj[0] += nx/nn * scale;
            env->obj[1] += ny/nn * scale;
            env->obj[2] += nz/nn * scale;
            // clamp to workspace and above floor
            for (int k=0;k<3;k++) env->obj[k] = clampf(env->obj[k], env->ws_min[k], env->ws_max[k]);
            if (env->obj[2] < env->obj_radius) env->obj[2] = env->obj_radius;
        }
    }

    // Distance deltas for shaping
    float dxg = env->obj[0]-env->goal[0];
    float dyg = env->obj[1]-env->goal[1];
    float dzg = env->obj[2]-env->goal[2];
    float obj_to_goal = sqrtf(dxg*dxg + dyg*dyg + dzg*dzg);
    float dxeo2 = env->ee[0]-env->obj[0];
    float dyeo2 = env->ee[1]-env->obj[1];
    float dzeo2 = env->ee[2]-env->obj[2];
    float ee_to_obj = sqrtf(dxeo2*dxeo2 + dyeo2*dyeo2 + dzeo2*dzeo2);

    float approach_obj = env->prev_ee_to_obj - ee_to_obj; if (approach_obj < 0) approach_obj = 0.0f;
    float approach_goal = env->grasped ? (env->prev_obj_to_goal - obj_to_goal) : 0.0f; if (approach_goal < 0) approach_goal = 0.0f;

    // Action penalty for smoothness
    float l2 = 0.0f; for(int i=0;i<n;i++){ float v = env->actions[i]; l2 += v*v; }
    float act_cost = env->action_penalty * sqrtf(l2);

    // Success conditions: object inside goal AABB and released
    int in_box = (fabsf(dxg) <= env->box_half && fabsf(dyg) <= env->box_half && env->obj[2] <= env->box_height + env->obj_radius*1.5f);

    float reward = 0.0f;
    reward += env->w_approach_obj * approach_obj;
    reward += env->w_approach_goal * approach_goal;
    reward -= act_cost;

    // Grasp bonus on transition and event counting
    if (!env->was_grasped && env->grasped){
        reward += env->bonus_grasp;
        env->total_grasps += 1;
    }
    env->was_grasped = env->grasped;

    // Place success and idle success
    env->terminals[0] = 0;
    if (in_box && !env->grasped){
        env->total_places += 1;
        if (env->t == 0 && sqrtf(l2) < 1e-6f){
            reward += env->bonus_idle; // already solved, don't move
        } else {
            reward += env->bonus_place;
        }
        env->terminals[0] = 1;
    }

    env->rewards[0] = reward;
    // update previous distances for next step
    env->prev_ee_to_obj = ee_to_obj;
    env->prev_obj_to_goal = obj_to_goal;
    env->t += 1;
    if (env->t >= env->max_steps) {
        env->terminals[0] = 1;
    }

    env->log.episode_return += reward;
    env->log.episode_length += 1.0f;
    // Log per-step distance to goal and success events (1.0 on successful place)
    env->log.dist_to_goal += obj_to_goal;
    env->log.success_rate += (in_box && !env->grasped) ? 1.0f : 0.0f;

    if (env->terminals[0]){
        env->log.n += 1.0f;
        env->log.total_grasps += (float)env->total_grasps;
        env->log.total_places += (float)env->total_places;
        c_reset(env);
    }

    write_obs(env);
}

/* c_render and c_close implemented below with Raylib viewer */

// --- Minimal Raylib viewer ---
typedef struct {
    Camera3D camera;
    int width;
    int height;
} SoArmClient;

static inline void ensure_client(SoArm* env){
    if (env->client != NULL) return;
    if (!IsWindowReady()){
        InitWindow(1080, 720, "PufferLib SO-ARM (C-native)");
        SetTargetFPS(60);
    }
    SoArmClient* c = (SoArmClient*)calloc(1, sizeof(SoArmClient));
    c->width = 1080; c->height = 720;
    c->camera.position = (Vector3){0.8f, 0.8f, 0.6f};
    c->camera.target   = (Vector3){0.0f, 0.0f, 0.2f};
    c->camera.up       = (Vector3){0.0f, 0.0f, 1.0f};
    c->camera.fovy     = 45.0f;
    c->camera.projection = CAMERA_PERSPECTIVE;
    env->client = c;
}

static inline void compute_link_positions(const SoArm* env, const float* q, float* out_xyz /* (dof+1)*3 */){
    // Base at origin
    out_xyz[0] = out_xyz[1] = out_xyz[2] = 0.0f;
    float T[16] = {1,0,0,0,
                   0,1,0,0,
                   0,0,1,0,
                   0,0,0,1};
    float A[16], C[16];
    for (int i=0;i<env->dof;i++){
        dh_A(env->dh_a[i], env->dh_alpha[i], env->dh_d[i], q[i], A);
        mat4_mul(T, A, C);
        memcpy(T, C, sizeof(float)*16);
        out_xyz[(i+1)*3 + 0] = T[3];
        out_xyz[(i+1)*3 + 1] = T[7];
        out_xyz[(i+1)*3 + 2] = T[11];
    }
}

static inline void draw_arm_links(const SoArm* env, const float* pts){
    Color link = (Color){0, 187, 187, 255};
    for (int i=0;i<env->dof;i++){
        Vector3 a = (Vector3){pts[i*3+0], pts[i*3+1], pts[i*3+2]};
        Vector3 b = (Vector3){pts[(i+1)*3+0], pts[(i+1)*3+1], pts[(i+1)*3+2]};
        DrawCylinderEx(a, b, 0.01f, 0.01f, 10, link);
        DrawSphere(a, 0.015f, link);
    }
}

static inline void c_render(SoArm* env){
    ensure_client(env);
    SoArmClient* c = (SoArmClient*)env->client;

    float pts[(SOARM_MAX_DOF+1)*3];
    compute_link_positions(env, env->q, pts);
    // Also compute full EE transform for orientation-based gripper drawing
    float T[16] = {1,0,0,0,
                   0,1,0,0,
                   0,0,1,0,
                   0,0,0,1};
    float A[16], C[16];
    for (int i=0;i<env->dof;i++){
        dh_A(env->dh_a[i], env->dh_alpha[i], env->dh_d[i], env->q[i], A);
        mat4_mul(T, A, C);
        memcpy(T, C, sizeof(float)*16);
    }

    BeginDrawing();
    ClearBackground((Color){6,24,24,255});
    BeginMode3D(c->camera);
    // ground grid
    for (int i=-5;i<=5;i++){
        DrawLine3D((Vector3){i*0.1f, -0.5f, 0.0f}, (Vector3){i*0.1f, 0.5f, 0.0f}, (Color){18,72,72,255});
        DrawLine3D((Vector3){-0.5f, i*0.1f, 0.0f}, (Vector3){0.5f, i*0.1f, 0.0f}, (Color){18,72,72,255});
    }
    // links
    draw_arm_links(env, pts);
    // ee/object/goal and simple bin (AABB)
    if (env->ee_radius > 0.0f) {
        DrawSphere((Vector3){env->ee[0], env->ee[1], env->ee[2]}, env->ee_radius, (Color){241,241,241,200});
    }
    // Gripper as two pads offset along EE local +Y with opening = jaw_width
    float spread = 0.5f * ((env->jaw_width > 0.0f) ? env->jaw_width : 0.03f);
    // EE local axes are columns of R (row-major layout)
    Vector3 yaxis = (Vector3){T[1], T[5], T[9]};
    Vector3 xaxis = (Vector3){T[0], T[4], T[8]};
    // Normalize axes
    float yn = sqrtf(yaxis.x*yaxis.x + yaxis.y*yaxis.y + yaxis.z*yaxis.z);
    float xn = sqrtf(xaxis.x*xaxis.x + xaxis.y*xaxis.y + xaxis.z*xaxis.z);
    if (yn < 1e-6f){ yaxis = (Vector3){0,1,0}; yn = 1.0f; }
    if (xn < 1e-6f){ xaxis = (Vector3){1,0,0}; xn = 1.0f; }
    Vector3 base = (Vector3){env->ee[0], env->ee[1], env->ee[2]};
    // Push pads slightly forward along +X to avoid overlap with EE sphere
    float pad_forward = 0.012f;
    Vector3 forward = (Vector3){ pad_forward*(xaxis.x/xn), pad_forward*(xaxis.y/xn), pad_forward*(xaxis.z/xn)};
    Vector3 gL = (Vector3){ base.x + forward.x - spread*(yaxis.x/yn), base.y + forward.y - spread*(yaxis.y/yn), base.z + forward.z - spread*(yaxis.z/yn)};
    Vector3 gR = (Vector3){ base.x + forward.x + spread*(yaxis.x/yn), base.y + forward.y + spread*(yaxis.y/yn), base.z + forward.z + spread*(yaxis.z/yn)};
    DrawCube(gL, 0.016f, 0.006f, 0.02f, (Color){220,220,220,255});
    DrawCube(gR, 0.016f, 0.006f, 0.02f, (Color){220,220,220,255});

    // Object as a red ball to fetch
    DrawSphere((Vector3){env->obj[0], env->obj[1], env->obj[2]}, env->obj_radius, (Color){187,0,0,255});
    // Goal bin as shallow 3D box (floor + 4 thin walls)
    float base_z = 0.005f;
    DrawCubeV((Vector3){env->goal[0], env->goal[1], base_z}, (Vector3){env->box_half*2.0f, env->box_half*2.0f, 0.01f}, (Color){0,187,187,80});
    float t = 0.005f; // wall thickness
    float h = env->box_height;
    // +X wall
    DrawCubeV((Vector3){env->goal[0]+env->box_half, env->goal[1], h*0.5f}, (Vector3){t, env->box_half*2.0f, h}, (Color){0,187,187,120});
    // -X wall
    DrawCubeV((Vector3){env->goal[0]-env->box_half, env->goal[1], h*0.5f}, (Vector3){t, env->box_half*2.0f, h}, (Color){0,187,187,120});
    // +Y wall
    DrawCubeV((Vector3){env->goal[0], env->goal[1]+env->box_half, h*0.5f}, (Vector3){env->box_half*2.0f, t, h}, (Color){0,187,187,120});
    // -Y wall
    DrawCubeV((Vector3){env->goal[0], env->goal[1]-env->box_half, h*0.5f}, (Vector3){env->box_half*2.0f, t, h}, (Color){0,187,187,120});
    EndMode3D();

    char buf[128];
    float dx = env->obj[0]-env->goal[0];
    float dy = env->obj[1]-env->goal[1];
    float dz = env->obj[2]-env->goal[2];
    float dist = sqrtf(dx*dx+dy*dy+dz*dz);
    snprintf(buf, sizeof(buf), "t=%d  dist=%.3f  grasped=%d  g=%d  p=%d", env->t, dist, (int)env->grasped, env->total_grasps, env->total_places);
    DrawText(buf, 10, 10, 20, (Color){241,241,241,255});
    EndDrawing();
}

static inline void c_close(SoArm* env){
    if (IsWindowReady()) CloseWindow();
    if (env->client){ free(env->client); env->client = NULL; }
}


