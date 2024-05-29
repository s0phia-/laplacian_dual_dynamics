import matplotlib as plt
import plotnine as p9
import pandas as pd
import numpy as np


class Plot:
    def __init__(self, data_source, data_source_2=None):
        self.data = pd.read_csv(data_source)
        if data_source_2 is not None:
            self.data_2 = pd.read_csv(data_source_2)
        else:
            self.data_2 = None

    def plot_scatter(self):
        p = (p9.ggplot(data=self.data,
                       mapping=p9.aes(x='steps',
                                      y='avg_return',
                                      color='datapoints'))
             + p9.geom_point()
             + p9.theme_bw()
             + p9.labs(color='Data points seen', x="Steps", y="Return")
             )
        print(p)

    def plot_two(self):
        data1 = self.data.copy()
        data2 = self.data_2.copy()
        data1["agent"] = "PVF"
        data2["agent"] = "Q learning"
        data = pd.concat([data1, data2])
        p = (p9.ggplot(data=data,
                       mapping=p9.aes(x='steps',
                                      y='avg_return',
                                      color='agent',
                                      alpha='datapoints'))
             + p9.geom_point()
             + p9.scale_alpha_continuous(range=[0.1, 1])
             + p9.theme_bw()
             + p9.labs(color='Data points seen', x="Steps", y="Return")
             )
        print(p)
