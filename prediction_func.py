from activation_fuc import sigmoid as sg


def prediction_function(age, affordability):
    w1 = 5.060867
    w2 = 1.4086502
    intercept = -2.9137027
    weighted_sum = w1*age + w2*affordability + intercept
    return sg(weighted_sum)


print(prediction_function(.18,1))
