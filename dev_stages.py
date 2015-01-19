# -*- coding: utf-8 -*-

# Devlopmental stages as defined in Kang
# Period        Description     Age
# M, postnatal months; PCW, post-conceptional weeks; Y, postnatal years.
#
# 1     Embryonic       4 PCW <= Age < 8 PCW
# 2     Early fetal     8 PCW <= Age < 10 PCW
# 3     Early fetal     10 PCW <= Age < 13 PCW
# 4     Early mid-fetal 13 PCW <= Age < 16 PCW
# 5     Early mid-fetal 16 PCW <= Age < 19 PCW
# 6     Late mid-fetal  19 PCW <= Age < 24 PCW
# 7     Late fetal      24 PCW <= Age <38 PCW
# 8     Neonatal and early infancy      0 M (birth) <= Age <6 M
# 9     Late infancy    6 M <= Age < 12 M
# 10    Early childhood 1 Y <= Age <6 Y
# 11    Middle and late childhood       6 Y <= Age < 12 Y
# 12    Adolescence     12 Y <= Age < 20 Y
# 13    Young adulthood 20 Y <= Age < 40 Y
# 14    Middle adulthood        40 Y <= Age < 60 Y
# 15    Late adulthood  60 Y <= Age

def PCW(x):
    """convert post-conceptional weeks => postnatal years"""
    return (-38.0 + x) * 7 / 365
    
def M(x):
    """convert postnatal months => postnatal years"""
    return x / 12.0

class DevStage(object):
    def __init__(self, num, name, short_name, from_age, to_age):
        self.num = num
        self.name = name
        self._short_name = short_name
        self.from_age = float(from_age)
        self.to_age = float(to_age)
        
    @property
    def short_name(self):
        return self._short_name #+ str(self.num)
        
    @property
    def central_age(self):
        return (self.from_age + self.to_age) / 2
        
    def scaled(self, scaler):
        import scalers
        scaler = scalers.unify(scaler) # handle None
        return DevStage(
            num = self.num,
            name = self.name,
            short_name = self.short_name,
            from_age = scaler.scale(self.from_age),
            to_age = scaler.scale(self.to_age),
        )

dev_stages = [
    DevStage(1,'Embryonic', 'E', PCW(4), PCW(8)),
    DevStage(2,'Early fetal', 'EF', PCW(8), PCW(13)), # EF2 + EMF3
    DevStage(4,'Early mid-fetal', 'EMF', PCW(13), PCW(19)), # EMF4 + EMF5
    DevStage(6,'Late mid-fetal', 'LMF', PCW(19), PCW(24)),
    DevStage(7,'Late fetal', 'LF', PCW(24), PCW(38)),
    DevStage(8,'Neonatal and early infancy', 'EI', M(0), M(6)),
    DevStage(9,'Late infancy', 'LI', M(6), M(12)),
    DevStage(10,'Early childhood', 'EC', 1, 6),
    DevStage(11,'Middle and late childhood', 'MLC', 6, 12),
    DevStage(12,'Adolescence', 'Adol', 12, 20),
    DevStage(13,'Young adulthood', 'YA', 20, 40),
    DevStage(14,'Middle adulthood', 'MA', 40, 60),
    DevStage(15,'Late adulthood', 'LA', 60, 80),
]

def get_stage_by_name(name, b_short=False):
    def name_getter(stage): return stage.short_name if b_short else stage.name
    for stage in dev_stages:
        if name_getter(stage) == name:
            return stage
    else:
        raise AssertionError('Development stage {} not found'.format(name))

def map_age_to_stage(age):
    for i,stage in enumerate(dev_stages):
        if age < stage.to_age:
            return stage
    else:
        return dev_stages[-1]
