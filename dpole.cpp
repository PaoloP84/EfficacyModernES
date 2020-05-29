/*
 *
 * Double-pole balancing problem
 *
 * This file implements the long-poles version only
 * (i.e., where the ratio between pole dimensions is 2 to 1).
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "dpole.h"

// Ranges for state variables
double stateRanges[NUM_STATES];

// Pointer to the observation
double* cobservation;
// Pointer to the action
double* caction;
// Pointer to termination flag
double* cdone;
// Pointer to world objects to be rendered
double* dobjects;

Problem::Problem()
{
    // Ranges for state variables are defined according to Pagliuca, Milano and Nolfi (2018)
    stateRanges[0] = 1.944;
    stateRanges[1] = 1.215;
    stateRanges[2] = 0.10472;
    stateRanges[3] = 0.135088;
    stateRanges[4] = stateRanges[2];
    stateRanges[5] = stateRanges[3];
    // Allocate memory for state variables
    m_state = new double[NUM_STATES];
    // Set num_inputs, num_outputs and action range [-1,1]
    m_ninputs = NUM_INPUTS;
    m_noutputs = 1;
    m_low = -1.0;
    m_high = 1.0;
    m_rng = new RandomGenerator(time(NULL));
}

Problem::~Problem()
{
}

/*
 * Check whether the cart is located within the 4.8m area
 * and whether the angle of the poles does not exceed 36 degrees
 */
bool Problem::outsideBounds()
{
    return
    m_state[0] < -TRACK_EDGE          ||
    m_state[0] > TRACK_EDGE           ||
    m_state[2] < -THIRTY_SIX_DEGREES  ||
    m_state[2] > THIRTY_SIX_DEGREES   ||
    m_state[4] < -THIRTY_SIX_DEGREES  ||
    m_state[4] > THIRTY_SIX_DEGREES;
}

void Problem::doStep(double action, double *st, double *derivs)
{
    double force, costheta_1, costheta_2, sintheta_1, sintheta_2, gsintheta_1, gsintheta_2, temp_1, temp_2, ml_1, ml_2, fi_1, fi_2, mi_1, mi_2;

    force =  (action - 0.5) * FORCE_MAG * 2;
    costheta_1 = cos(st[2]);
    sintheta_1 = sin(st[2]);
    gsintheta_1 = GRAVITY * sintheta_1;
    costheta_2 = cos(st[4]);
    sintheta_2 = sin(st[4]);
    gsintheta_2 = GRAVITY * sintheta_2;

    ml_1 = LENGTH_1 * MASSPOLE_1;
    ml_2 = LENGTH_2 * MASSPOLE_2;
    temp_1 = MUP * st[3] / ml_1;
    temp_2 = MUP * st[5] / ml_2;
    fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) +
           (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
    fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) +
           (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
    mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));
    mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));

    derivs[1] = (force + fi_1 + fi_2)
                / (mi_1 + mi_2 + MASSCART);

    derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1)
                / LENGTH_1;
    derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2)
                / LENGTH_2;
}

void Problem::rk4(double f, double *y, double *dydx, double *yout)
{
    int i;

    double hh, h6, dym[6], dyt[6], yt[6];

    hh = TAU * 0.5;
    h6 = TAU / 6.0;
    for (i = 0; i <= 5; i++)
        yt[i] = y[i] + hh * dydx[i];
    doStep(f, yt, dyt);
    dyt[0] = yt[1];
    dyt[2] = yt[3];
    dyt[4] = yt[5];
    for (i = 0; i <= 5; i++)
        yt[i] = y[i] + hh * dyt[i];
    doStep(f, yt, dym);
    dym[0] = yt[1];
    dym[2] = yt[3];
    dym[4] = yt[5];
    for (i = 0; i <= 5; i++) {
        yt[i] = y[i] + TAU * dym[i];
        dym[i] += dyt[i];
    }
    doStep(f, yt, dyt);
    dyt[0] = yt[1];
    dyt[2] = yt[3];
    dyt[4] = yt[5];
    for (i = 0;i <= 5; i++)
        yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
}

/*
 * Apply the chosen torque and compute the new state of the cart and of the poles
 * by calling the functions step() and rk4()
 */
void Problem::performAction(double output)
{
    int i;
    double  dydx[6];

    const bool RK4 = true; //Use the Runge-Kutta 4th order integration method
    const double EULER_TAU = TAU / 4;

    /*--- Apply action to the simulated cart-pole ---*/

    if (RK4)
    {
        for (i = 0; i < 2; ++i)
        {
            dydx[0] = m_state[1];
            dydx[2] = m_state[3];
            dydx[4] = m_state[5];
            doStep(output, m_state, dydx);
            rk4(output, m_state, dydx, m_state);
        }
    }
    else
    {
        for (i = 0; i < 8; ++i)
        {
            doStep(output, m_state, dydx);
            m_state[0] += EULER_TAU * dydx[0];
            m_state[1] += EULER_TAU * dydx[1];
            m_state[2] += EULER_TAU * dydx[2];
            m_state[3] += EULER_TAU * dydx[3];
            m_state[4] += EULER_TAU * dydx[4];
            m_state[5] += EULER_TAU * dydx[5];
        }
    }
}

void Problem::seed(int s)
{
    // Set the seed
    m_rng->setSeed(s);
}

void Problem::getObs()
{
    cobservation[0] = m_state[0] / 4.8;
    cobservation[1] = m_state[2] / 0.52;
    cobservation[2] = m_state[4] / 0.52;
}

void Problem::copyObs(double* observation)
{
    cobservation = observation;
}
    
void Problem::copyAct(double* action)
{
    caction = action;
}

void Problem::copyDone(double* done)
{
    cdone = done;
}

void Problem::copyDobj(double* objs)
{
    dobjects = objs;
}

void Problem::reset()
{
    int s;
    // Initialize state
    for (s = 0; s < NUM_STATES; s++)
        m_state[s] = m_rng->getDouble(-stateRanges[s], stateRanges[s]);
    // Get observations
    getObs();
}

double Problem::step()
{
    double reward;
    // Perform the action
    performAction(*caction);
    // Get observations
    getObs();
    // terminate prematurely in case of failure
    *cdone = 0.0;
    if (outsideBounds())
        *cdone = 1.0;
    // Return reward depending on whether or not the poles are still balanced
    if (*cdone == 0.0)
        reward = 1.0;
    else
        reward = 0.0;
    return reward;
}

void Problem::close()
{
    //printf("close() not implemented\n");
}

/*
 * create the list objects to be rendered graphically (i.e., the cart, the poles and the track)
 */
void Problem::render()
{
    int i;
    int c;

    c = 0;
    // track
    dobjects[c] = 2.0;
    dobjects[c+1] = 200.0 - TRACK_EDGE * 50.0;
    dobjects[c+2] = 100.0;
    dobjects[c+3] = 200.0 + TRACK_EDGE * 50.0;
    dobjects[c+4] = 100.0;
    dobjects[c+5] = 0.0;
    dobjects[c+6] = 0.0;
    dobjects[c+7] = 0.0;
    dobjects[c+8] = 0.0;
    dobjects[c+9] = 0.0;
    c += 10;
    // track edges
    for (i = 0; i < 2; i++)
    {
        dobjects[c] = 2.0;
        if (i == 0)
        {
            // left edge
            dobjects[c+1] = 200.0 - TRACK_EDGE * 50.0;
            dobjects[c+2] = 100.0;
            dobjects[c+3] = 200.0 - TRACK_EDGE * 50.0;
            dobjects[c+4] = 125.0;
        }
        else
        {
            // right edge
            dobjects[c+1] = 200.0 + TRACK_EDGE * 50.0;
            dobjects[c+2] = 100.0;
            dobjects[c+3] = 200.0 + TRACK_EDGE * 50.0;
            dobjects[c+4] = 125.0;
        }
        dobjects[c+5] = 0.0;
        dobjects[c+6] = 0.0;
        dobjects[c+7] = 0.0;
        dobjects[c+8] = 0.0;
        dobjects[c+9] = 0.0;
        c += 10;
    }
    // cart
    // body
    dobjects[c] = 4.0;
    dobjects[c+1] = 200.0 + (m_state[0] - 0.75) * 50.0;
    dobjects[c+2] = 125.0;
    dobjects[c+3] = 200.0 + (m_state[0] + 0.75) * 50.0;
    dobjects[c+4] = 150.0;
    dobjects[c+5] = 255.0;
    dobjects[c+6] = 0.0;
    dobjects[c+7] = 0.0;
    dobjects[c+8] = 0.0;
    dobjects[c+9] = 0.0;
    c += 10;
    // wheels
    for (i = 0; i < 2; i++)
    {
        dobjects[c] = 3.0;
        if (i == 0)
            // left wheel
            dobjects[c+1] = 200.0 + (m_state[0] - 0.375) * 50.0;
        else
            // right wheel
            dobjects[c+1] = 200.0 + (m_state[0] + 0.375) * 50.0;
        dobjects[c+2] = 112.5;
        dobjects[c+3] = 12.5;//6.25;
        dobjects[c+4] = 0.0;
        dobjects[c+5] = 0.0;
        dobjects[c+6] = 0.0;
        dobjects[c+7] = 0.0;
        dobjects[c+8] = 0.0;
        dobjects[c+9] = 0.0;
        c += 10;
    }
    // poles
    for(i = 0; i < 2; i++)
    {
        dobjects[c] = 2.0;
        if (i == 0)
        {
            dobjects[c+1] = 200.0 + (m_state[0] - 0.25) * 50.0;
            dobjects[c+2] = 150.0;
            dobjects[c+3] = 200.0 + (m_state[0] - 0.25 + (LENGTH_1 * 5.0 * cos(M_PI / 2.0 + m_state[2]))) * 50.0;
            dobjects[c+4] = 150.0 + (LENGTH_1 * 5.0 * sin(M_PI / 2.0 + m_state[2])) * 50.0;
        }
        else
        {
            dobjects[c+1] = 200.0 + (m_state[0] + 0.25) * 50.0;
            dobjects[c+2] = 150.0;
            dobjects[c+3] = 200.0 + (m_state[0] + 0.25 + (LENGTH_2 * 5.0 * cos(M_PI / 2.0 + m_state[4]))) * 50.0;
            dobjects[c+4] = 150.0 + (LENGTH_2 * 5.0 * sin(M_PI / 2.0 + m_state[4])) * 50.0;
        }
        dobjects[c+5] = 0.0;
        dobjects[c+6] = 0.0;
        dobjects[c+7] = 255.0;
        dobjects[c+8] = 0.0;
        dobjects[c+9] = 0.0;
        c += 10;
    }
    dobjects[c] = 0.0;
}

