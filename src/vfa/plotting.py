import matplotlib as plt
import plotnine as p9
import pandas as pd
import numpy as np


class Plot:
    def __init__(self, data_source):
        self.data = pd.read_csv(data_source)

    def plot_scatter(self):
        p = (p9.ggplot(data=self.data,
                       mapping=p9.aes(x='steps',
                                      y='avg_return',
                                      color='datapoints'))
             + p9.geom_point()
             + p9.theme_bw()
             + p9.labs(color='Data points seen', x="Steps", y="Log Return")
             )
        print(p)


Plot('../../out.csv').plot_scatter()
