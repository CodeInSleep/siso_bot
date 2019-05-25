import matplotlib.pyplot as plt
import pdb

def plot_vec(df):
    dim = df.shape[1]
    assert (dim == 3), "not 3 dimensions"

    series = [df[col].values for col in df.columns]
    xs, ys, zs = series
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs)

def plot_df(df, plot_name, fields = []):
    # plotting inputs
    plt.title = plot_name
   
    if len(fields) == 3:
        plot_vec(df[fields])
    elif len(fields) == 2:
        #plot_vec(df[fields])
        plt.plot(df[fields[0]], df[fields[1]])
    else:
        plot = plt.figure()
        for f in fields:
            if f in df:
                df[f].plot()
            else:
                print('field %s does not exist in df' % f)

