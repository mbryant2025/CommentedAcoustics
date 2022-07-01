# to adjust this script to different noise environment,
# change fft_w_size, large_window_portion, and invalidation requirement in get_pdiff

import pandas
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

# Saleae sampling frequency
SAMPLING_FREQUENCY = 625_000

# Number of samples taken during a 0.004s ping
ping_samples = SAMPLING_FREQUENCY * 0.004

# Pinger (target) frequency
PINGER_FREQUENCY = 40_000

# Speed of sound in water in m/s
SOUND_VELO = 1511.5

# Nipple distance between hydrophones
HYDROPHONE_SPACING = 0.0115

# Phase difference between hydrophones = 2pi * (HYDROPHONE_SPACING / wavelength)
# Michael's note: this is unused in this script 
phase_diff = np.pi*2 * HYDROPHONE_SPACING * PINGER_FREQUENCY / SOUND_VELO

fft_w_size = 125

check_len = 20

# Number of chunks to split phase difference data into
large_window_portion = 5

guess = (0, 0, -10)

# Represents relative position of hydrophones
hydrophone_positions = [np.array([0, 0, 0]),
      np.array([0, -HYDROPHONE_SPACING, 0]),
      np.array([-HYDROPHONE_SPACING, 0, 0]),
      np.array([-HYDROPHONE_SPACING, -HYDROPHONE_SPACING, 0])]

# Potential future configuration for tetrahedral geometry
# hydrophone_positions = [np.array([0, 0, 0]),
#       np.array([-HYDROPHONE_SPACING/2, -np.sqrt(3)*HYDROPHONE_SPACING/2, 0]),
#       np.array([-HYDROPHONE_SPACING, 0, 0]),
#       np.array([-HYDROPHONE_SPACING/2, -np.sqrt(3)*HYDROPHONE_SPACING/4, -3*HYDROPHONE_SPACING/4])]


# Returns average phase difference in window with least variance
# Phase difference is measured between channels parr1 and parr2 
# Only consideres data in parr1 and parr2 between indicies start and end
# Window size is determined from large_window_portion
def get_pdiff(parr1, parr2, start, end):

    #Phase difference between channels
    phase_diff = np.subtract(parr2, parr1)

    #Phase difference between channels within window and corrected for differences greater than a full period
    phase_diff_corr = correct_phase(phase_diff[start:end])

    # List of variances of phase differences within phase_diff_corr
    # Each index in var represents the variance of the chunk starting at that index
    # Chunk size is the second parameter of variance_list; the number of chunks is large_window_portion
    # var is of length len(phase_diff_corr) - the size of the chunk 
    var = variance_list(phase_diff_corr, int(len(phase_diff_corr)//large_window_portion))

    # Index of minimum variance
    phase_start = np.argmin(var)

    # THIS REQUIREMENT CAN BE LOSEN IF TOO MANY PINGS ARE INVALID, YET MIGHT LEAD TO INACCURATE RESULT. Change 2 to 1.5.
    # If the minimum variance is in the second half of the window return None
    # Second half contains lots of noise from reflections, so it should not be used
    if phase_start > len(phase_diff_corr)/2:
        return None

    # Mark end of phase difference window of least variance 
    phase_end = phase_start + int(len(phase_diff_corr)//large_window_portion)

    #Return average phase difference in window of least variance
    return np.mean(phase_diff_corr[phase_start:phase_end])

# Offsets phase difference by 2pi -- used in correct_phase
def reduce_phase(phase):
    return -phase/abs(phase)*(2*np.pi-abs(phase))

# If phase difference is greater than pi in either direction, correct it by offsetting by 2pi
def correct_phase(arr):
    return [reduce_phase(phase) if abs(phase) > np.pi else phase for phase in arr]

def fft_sw(xn, freq):
    n = np.arange(len(xn))
    exp = np.exp(-1j*freq/SAMPLING_FREQUENCY*2*np.pi*n)
    return np.dot(xn, exp)

def get_alist(a, n):
    weights = np.repeat(1.0, n)/n
    return np.convolve(a, weights, 'valid')

def moving_average_max(a, n = int(ping_samples/fft_w_size)):
    return np.argmax(get_alist(a, n))

def fft(xn, PINGER_FREQUENCY, w_size):
    ft = []
    for i in range(int(len(xn)//w_size)):
        xn_s = xn[i*w_size:(i+1)*w_size]
        ft.append(fft_sw(xn_s, PINGER_FREQUENCY))
    return np.angle(ft), np.absolute(ft)

# Returns a list of variances in arr of chunks of size window
def variance_list(arr, window):
    return [np.var(arr[i:i+window]) for i in range(len(arr) - window + 1)]

def apply_to_pairs(fn, arr):
    return [fn(arr[0], arr[1]), fn(arr[0], arr[2]), fn(arr[2], arr[3])]

def diff_equation(hp_a, hp_b, target, t_diff):
    return (np.linalg.norm(target-hp_a) - np.linalg.norm(target-hp_b)) - t_diff

def system(target, *data):
    diff_12, diff_13, diff_34 = data
    return (diff_equation(hydrophone_positions[0], hydrophone_positions[1], target, diff_12),
            diff_equation(hydrophone_positions[0], hydrophone_positions[2], target, diff_13),
            diff_equation(hydrophone_positions[2], hydrophone_positions[3], target, diff_34))

def solver(guess, diff):
    data_val = (diff[0], diff[1], diff[2])
    return fsolve(system, guess, args=data_val)

def data_to_pdiff(data):
    plist_all, mlist_all = zip(*[fft(d, PINGER_FREQUENCY, fft_w_size) for d in data])
    mlist = np.sum(mlist_all, axis=0)
    mag_start = moving_average_max(mlist)
    mag_end = mag_start + int(ping_samples//fft_w_size)
    return apply_to_pairs(lambda p1, p2: get_pdiff(p1, p2, mag_start, mag_end), plist_all)

def read_data(filepath):
    df = pandas.read_csv(filepath, skiprows=[1], skipinitialspace=True)
    print("running ", filepath)
    return [df["Channel 0"].tolist(), df["Channel 1"].tolist(), df["Channel 2"].tolist(), df["Channel 3"].tolist()]

def split_data(data):
    return [data[0:int(len(data)/2)], data[int(len(data)/2):len(data)]]

def check_angle(hz, vt):
    pinger = np.array([check_len*np.cos(vt)*np.cos(hz), check_len*np.cos(vt)*np.sin(hz), -check_len*np.sin(vt)])
    dis = [np.linalg.norm(pinger-hp_val) for hp_val in hydrophone_positions]
    return apply_to_pairs(lambda dis1, dis2: (dis1 - dis2)/SOUND_VELO*PINGER_FREQUENCY*2*np.pi, dis)

def plot_3d(ccwh, downv, ax):
    if abs(ccwh) < np.pi/2:
        xline = np.linspace(0, 10, 100)
    else:
        xline = np.linspace(-10, 0, 100)
    yline = xline*np.tan(ccwh)
    zline = -xline/np.cos(ccwh)*np.tan(downv)
    ax.plot3D(xline, yline, zline, 'gray')

def plot(ccwha, downva, count, actual):
    print("\nsuccess count: ", count, "\n")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D([guess[0]], [guess[1]], [guess[2]], color="b")
    # ax.scatter3D([val[0] for val in actual], [val[1] for val in actual], [val[2] for val in actual], color="r")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    for i in range(len(ccwha)):
        plot_3d(ccwha[i], downva[i], ax)
    plt.show()

def final_hz_angle(hz_arr):
    quadrant = [[], [], [], []]
    quadrant[0] = [hz for hz in hz_arr if 0 <= hz <= np.pi/2]
    quadrant[1] = [hz for hz in hz_arr if hz > np.pi/2]
    quadrant[2] = [hz for hz in hz_arr if hz <= -np.pi/2]
    quadrant[3] = [hz for hz in hz_arr if 0 > hz > -np.pi/2]
    max_len = np.max([len(each) for each in quadrant])
    ave = [q for q in quadrant if len(q) == max_len]
    return np.mean(ave)

def process_data(raw_data, if_double, actual, ccwha, downva, count):
    data = np.array([split_data(d) if if_double else [d] for d in raw_data])

    for j in range(len(data[0])):
        pdiff = data_to_pdiff(data[:, j])

        if None not in pdiff:
            count += 1
            diff = [(val / 2 / np.pi * SOUND_VELO / PINGER_FREQUENCY) for val in pdiff]

            ans = solver(guess, diff)
            x, y, z = ans[0], ans[1], ans[2]
            actual.append((x, y, z))
            print("initial guess", guess)
            print("x, y, z", actual[-1])

            # calculate angle
            ccwh = np.arctan2(y, x)
            ccwha.append(ccwh)
            print("horizontal angle", np.rad2deg(ccwh))
            downv = np.arctan2(-z, np.sqrt(x ** 2 + y ** 2))
            downva.append(downv)
            print("vertical downward angle", np.rad2deg(downv), "\n")

            # compare solver result with pdiff from data
            pd = check_angle(ccwh, downv)
            print("checked pd", pd[0], pd[1], pd[2])
            print("pdiff_12", pdiff[0], "pdiff_13", pdiff[1], "pdiff_34", pdiff[2], "\n")
        else:
            print("invalid\n")
    return actual, ccwha, downva, count

#Inputs file, signal parameters, and a guess and ouputs 
def cross_corr_func(filename, if_double, version, if_plot, samp_f=SAMPLING_FREQUENCY, tar_f=PINGER_FREQUENCY, guess_x=guess[0], guess_y=guess[1], guess_z=guess[2]):
    global SAMPLING_FREQUENCY, PINGER_FREQUENCY, guess
    SAMPLING_FREQUENCY = samp_f
    PINGER_FREQUENCY = tar_f
    filepath = filename
    guess = (guess_x, guess_y, guess_z)

    ccwha = []
    downva = []
    actual = []
    count = 0

    if version == 0:
        raw_data = read_data(filepath)
        actual, ccwha, downva, count = process_data(raw_data, if_double, actual, ccwha, downva, count)

    for i in range(version):
        raw_data = read_data(filepath.replace(".csv", "("+str(i+1)+").csv"))
        actual, ccwha, downva, count = process_data(raw_data, if_double, actual, ccwha, downva, count)

    final_ccwh = final_hz_angle(ccwha)
    print("\n\nfinal horizontal angle", np.rad2deg(final_ccwh))

    if if_plot:
        plot(ccwha, downva, count, actual)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        cross_corr_func(sys.argv[1], True, 4, True, 625000, 40000, 0, 0, -10)
    else:
        try:
            cross_corr_func(sys.argv[1], sys.argv[2] == "True", int(sys.argv[3]), sys.argv[4] == "True", int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]))
        except:
            print("wrong input arguments")
