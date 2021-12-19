drop table if exists batter_count_with_date;
create table batter_count_with_date as
  select bc.*, g.local_date as local_date
  from batter_counts bc
  join game g
  on bc.game_id = g.game_id
  ORDER BY batter, local_date;

CREATE UNIQUE INDEX rolling_lookup_date_game_batter_id_idx ON batter_count_with_date (game_id, batter, local_date);
CREATE UNIQUE INDEX rolling_lookup_game_batter_id_idx ON batter_count_with_date (game_id, batter);
CREATE UNIQUE INDEX rolling_lookup_date_batter_id_idx ON batter_count_with_date (local_date, batter);
CREATE INDEX rolling_lookup_game_id_idx ON batter_count_with_date (game_id);
CREATE INDEX rolling_lookup_local_date_idx ON batter_count_with_date (local_date);
CREATE INDEX rolling_lookup_batter_idx ON batter_count_with_date (batter);

drop table if exists pitcher_count_with_date;
create table pitcher_count_with_date as
  select pc.*, g.local_date as local_date
  from pitcher_counts pc
  join game g
  on pc.game_id = g.game_id
  ORDER BY pitcher, local_date;

  CREATE UNIQUE INDEX rolling_lookup_date_game_pitcher_id_idx ON pitcher_count_with_date (game_id, pitcher, local_date);
  CREATE UNIQUE INDEX rolling_lookup_game_pitcher_id_idx ON pitcher_count_with_date (game_id, pitcher);
  CREATE UNIQUE INDEX rolling_lookup_date_pitcher_id_idx ON pitcher_count_with_date (local_date, pitcher);
  CREATE INDEX rolling_lookup_game_id_idx ON pitcher_count_with_date (game_id);
  CREATE INDEX rolling_lookup_local_date_idx ON pitcher_count_with_date (local_date);
  CREATE INDEX rolling_lookup_pitcher_idx ON pitcher_count_with_date (pitcher);

drop table if exists batterstat0;
create table batterstat0 as SELECT bc1.game_id, bc1.local_date,bc1.team_id,
bc1.hometeam,
AVG(bc2.hit) / nullif(AVG(bc2.atbat), 0) as ba,
AVG(bc2.home_run) AS HR,
AVG(bc2.walk) AS Walks,
AVG(bc2.Home_Run)/nullif(AVG(bc2.hit),0) AS hrh,
AVG(bc2.walk)/nullif(AVG(bc2.strikeout),0) as bbk,
(AVG(bc2.single)+AVG(2*bc2.double)+AVG(3*bc2.triple)+AVG(4*bc2.home_run))/nullif(AVG(bc2.atbat),0) as slg
FROM batter_count_with_date bc1
JOIN batter_count_with_date bc2 ON bc1.batter = bc2.batter AND bc2.local_date BETWEEN DATE_SUB(bc1.local_date, INTERVAL 6 DAY) AND DATE_SUB(bc1.local_date, INTERVAL 1 DAY)
WHERE bc1.hometeam=0
GROUP BY bc1.game_id, bc1.local_date, bc1.team_id, bc1.hometeam;

drop table if exists batterstat1;
create  table batterstat1 as SELECT bc1.game_id, bc1.local_date,bc1.team_id,
bc1.hometeam,
AVG(bc2.hit) / nullif(AVG(bc2.atbat), 0) as ba,
AVG(bc2.home_run) AS HR,
AVG(bc2.walk) AS Walks,
AVG(bc2.Home_Run)/nullif(AVG(bc2.hit),0) AS hrh,
AVG(bc2.walk)/nullif(AVG(bc2.strikeout),0) as bbk,
(AVG(bc2.single)+AVG(2*bc2.double)+AVG(3*bc2.triple)+AVG(4*bc2.home_run))/nullif(AVG(bc2.atbat),0) as slg
FROM batter_count_with_date bc1
JOIN batter_count_with_date bc2 ON bc1.batter = bc2.batter AND bc2.local_date BETWEEN DATE_SUB(bc1.local_date, INTERVAL 6 DAY) AND DATE_SUB(bc1.local_date, INTERVAL 1 DAY)
WHERE bc1.hometeam=1
GROUP BY bc1.game_id, bc1.local_date, bc1.team_id, bc1.hometeam;

drop table if exists batter_features;
create table batter_features as select ht.game_id,
ht.team_id as hometeam,
at.team_id as awayteam,
ht.ba - at.ba as ba_diff,
ht.hr - at.hr as hr_diff,
ht.walks - at.walks as walk_diff,
ht.hrh-at.hrh as hrh_diff,
ht.bbk-at.bbk as bbk_diff,
ht.slg-at.slg as slg_diff,
case
  when bs.winner_home_or_away = "H" then 1
  when bs.winner_home_or_away = "A" then  0
end as homewins
from batterstat1 ht
join batterstat0 at on ht.game_id = at.game_id
join boxscore bs on ht.game_id = bs.game_id;

drop table if exists spitcherstat0;
create table spitcherstat0 as SELECT pc1.game_id, pc1.local_date,pc1.team_id,
pc1.hometeam,
AVG(pc2.outsPlayed/3) as innings_pitched,
AVG(pc2.pitchesthrown) AS pitchesthrown,
AVG(pc2.hit) AS hits,
AVG(pc2.groundout) AS groundout,
AVG(pc2.walk) as walks,
AVG(pc2.pop_out) as popouts
FROM pitcher_count_with_date pc1
JOIN pitcher_count_with_date pc2 ON pc1.pitcher = pc2.pitcher AND pc2.local_date BETWEEN DATE_SUB(pc1.local_date, INTERVAL 101 DAY) AND DATE_SUB(pc1.local_date, INTERVAL 1 DAY)
WHERE pc1.hometeam=0 and pc1.startingpitcher=1
GROUP BY pc1.game_id, pc1.local_date, pc1.team_id, pc1.hometeam;

drop table if exists spitcherstat1;
create table spitcherstat1 as SELECT pc1.game_id, pc1.local_date,pc1.team_id,
pc1.hometeam,
AVG(pc2.outsPlayed/3) as innings_pitched,
AVG(pc2.pitchesthrown) AS pitchesthrown,
AVG(pc2.hit) AS hits,
AVG(pc2.groundout) AS groundout,
AVG(pc2.walk) as walks,
AVG(pc2.pop_out) as popouts
FROM pitcher_count_with_date pc1
JOIN pitcher_count_with_date pc2 ON pc1.pitcher = pc2.pitcher AND pc2.local_date BETWEEN DATE_SUB(pc1.local_date, INTERVAL 101 DAY) AND DATE_SUB(pc1.local_date, INTERVAL 1 DAY)
WHERE pc1.hometeam=1 and pc1.startingpitcher=1
GROUP BY pc1.game_id, pc1.local_date, pc1.team_id, pc1.hometeam;

drop table if exists spitcher_features;
create table spitcher_features as select ht.game_id,
ht.team_id as hometeam,
at.team_id as awayteam,
ht.innings_pitched - at.innings_pitched as ip_diff,
ht.pitchesthrown - at.pitchesthrown as pt_diff,
ht.hits - at.hits as hit_diff,
ht.groundout-at.groundout as gd_diff,
ht.walks-at.walks as w_diff,
ht.popouts-at.popouts as po_diff,
case
  when bs.winner_home_or_away = "H" then 1
  when bs.winner_home_or_away = "A" then  0
end as homewins
from spitcherstat1 ht
join spitcherstat0 at on ht.game_id = at.game_id
join boxscore bs on ht.game_id = bs.game_id;

drop table if exists travel;
create table travel as
  select pn1.game_id as current_game,
  pn1.home_away as current_game_ha,
  pn2.game_id as prior_game,
  pn2.home_away as prior_game_ha,
  pn1.team_id
  from team_game_prior_next pn1
join team_game_prior_next pn2
on pn1.prior_game_id=pn2.game_id
and pn1.team_id=pn2.team_id;

drop table if exists features;
create  table features as select b.game_id,
      dayofyear(g.local_date) as local_date,
      case when t.current_game_ha != t.prior_game_ha
        then 1
        else 0
        end as traveled_prior,
      b.hometeam,
      b.awayteam,
      b.homewins,
      b.ba_diff,
      b.hr_diff,
      b.walk_diff,
      b.hrh_diff,
      b.bbk_diff,
      b.slg_diff,
      p.ip_diff,
      p.pt_diff,
      p.hit_diff,
      p.gd_diff,
      p.w_diff,
      p.po_diff
      from batter_features b join spitcher_features p on b.game_id= p.game_id
      join game g on g.game_id = b.game_id
      join travel t on t.current_game = g.game_id;
