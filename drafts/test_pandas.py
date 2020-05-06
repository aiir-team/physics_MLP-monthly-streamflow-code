#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:42, 06/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import pandas as pd

temp = {'name': ['Raphael', 'Donatello'],
        'mask': ['red', 'purple'],
        'weapon': ['sai', 'bo staff']}
df = pd.DataFrame(temp, columns=['name', 'mask', 'weapon'])
df.to_csv('test.csv', index=False, header=True)

# event_saving = {"EventId": final_event_list2[:, 0], "EventTemplate": final_event_list2[:, 1]}
# df1 = DataFrame(event_saving, columns=["EventId", "EventTemplate"])
# df1.to_csv(path.join(output_dir, output3), index=False, header=True)

