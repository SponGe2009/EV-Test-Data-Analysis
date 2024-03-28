import numpy as np


def power_integration(power, start_time, end_time):
    integration_results = []
    for start, end in zip(start_time, end_time):
        # Filter the power for the current start and end time range
        filtered_power = power[(power[:, 0] >= start) & (power[:, 0] <= end)]

        # Compute the power integration for this time segment
        # Assuming linear trapezoidal integration, similar to the numpy trapz method
        if len(filtered_power) > 1:
            integrated_power = np.trapz(
                filtered_power[:, 1], filtered_power[:, 0])
        else:
            integrated_power = 0  # If there is only one point, we can't integrate

        integration_results.append(integrated_power / 3600)

    return integration_results


def power_integration(power, start_time, end_time):
    # Initialize an empty list to store the integration results for each segment
    integration_results = []

    for start, end in zip(start_time, end_time):
        # Filter the power for the current start and end time range
        filtered_power = power[(power[:, 0] >= start) & (power[:, 0] <= end)]

        # Initialize integration values for non-negative and negative parts
        integrated_power_positive = 0
        integrated_power_negative = 0

        # Compute the power integration for non-negative and negative parts separately
        if len(filtered_power) > 1:
            integrated_power = np.trapz(
                filtered_power[:, 1], filtered_power[:, 0])
            # Non-negative part
            positive_power = np.where(
                filtered_power[:, 1] > 0, filtered_power[:, 1], 0)
            integrated_power_positive = np.trapz(
                positive_power, filtered_power[:, 0])

            # Negative part
            negative_power = np.where(
                filtered_power[:, 1] < 0, filtered_power[:, 1], 0)
            integrated_power_negative = np.trapz(
                negative_power, filtered_power[:, 0])
        # Convert power integration from watt-seconds to watt-hours for consistency
        integration_results.append(
            [integrated_power / 3600, integrated_power_positive / 3600, integrated_power_negative / 3600])

    # Convert the results list to a NumPy ndarray
    return np.array(integration_results)
