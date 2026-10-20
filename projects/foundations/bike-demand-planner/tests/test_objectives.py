"""Independent loss arithmetic, finite differences, representation and fit contracts."""

import unittest

import numpy as np

from bike_demand_planner.features import encode_calendar
from bike_demand_planner.objectives import (LossConfig, apply_prediction_policy, calendar_representation,
                                           fit_objective_model, loss_values_and_slopes, objective_and_gradients)


def calendars():
    return np.asarray([encode_calendar(dict(hr=h, weekday=d, mnth=m, holiday=0, workingday=w))
                       for h,d,m,w in [(2,1,1,1),(8,1,1,1),(20,0,2,0),(23,6,12,0),(0,1,1,1)]])


class ObjectiveTests(unittest.TestCase):
    def test_known_losses_slopes_and_kink_policies(self):
        e = np.array([-14., -10., 0., 10., 14.])
        values, slopes = loss_values_and_slopes(e, LossConfig("huber", 10))
        np.testing.assert_array_equal(values, [90,50,0,50,90])
        np.testing.assert_array_equal(slopes, [-10,-10,0,10,10])
        values, slopes = loss_values_and_slopes(e, LossConfig("mae"))
        np.testing.assert_array_equal(values, [14,10,0,10,14])
        np.testing.assert_array_equal(slopes, [-1,-1,0,1,1])
        values, slopes = loss_values_and_slopes([-2,3], LossConfig())
        np.testing.assert_array_equal(values, [2,4.5]);np.testing.assert_array_equal(slopes, [-2,3])

    def test_all_gradients_against_central_differences(self):
        X=np.array([[1.,2.],[-2.,1.],[3.,-1.]])
        y=np.array([4.,-7.,9.]);theta=np.array([.4,-.7,1.2]);epsilon=1e-6
        for kind in ("half_mse","mae","huber"):
            config=LossConfig(kind,delta=2,l2=.3)
            _,dw,db=objective_and_gradients(X,y,theta[:-1],theta[-1],config)
            numerical=[]
            for j in range(3):
                plus,minus=theta.copy(),theta.copy();plus[j]+=epsilon;minus[j]-=epsilon
                numerical.append((objective_and_gradients(X,y,plus[:-1],plus[-1],config)[0]-
                                  objective_and_gradients(X,y,minus[:-1],minus[-1],config)[0])/(2*epsilon))
            np.testing.assert_allclose(np.append(dw,db),numerical,rtol=1e-7,atol=1e-7)

    def test_mean_reduction_penalty_normalization_and_unpenalized_bias(self):
        X=np.array([[1.,0.],[0.,1.]])
        y=X@np.array([2.,-3.])+100
        total,dw,db=objective_and_gradients(X,y,[2,-3],100,LossConfig(l2=.2))
        self.assertAlmostEqual(total,1.3);np.testing.assert_allclose(dw,[.4,-.6]);self.assertEqual(db,0)
        duplicate=objective_and_gradients(np.vstack((X,X)),np.tile(y,2),[2,-3],100,LossConfig(l2=.2))
        self.assertAlmostEqual(duplicate[0],total);np.testing.assert_allclose(duplicate[1],dw)

    def test_huber_caps_outlier_pull_and_is_not_normalized_smooth_l1(self):
        values,slopes=loss_values_and_slopes([-4,-400],LossConfig("huber",10))
        np.testing.assert_array_equal(slopes,[-4,-10]);np.testing.assert_array_equal(values,[8,3950])
        self.assertEqual(loss_values_and_slopes([-400],LossConfig("mae"))[1][0],-1)

    def test_calendar_structure_names_periods_and_interactions(self):
        X=calendars()
        cyclic,names=calendar_representation(X,"cyclic8")
        self.assertEqual(cyclic.shape,(5,8));self.assertEqual(len(names),8)
        self.assertAlmostEqual(cyclic[1,0],np.sqrt(3)/2);self.assertAlmostEqual(cyclic[1,1],-.5)
        self.assertAlmostEqual(cyclic[1,4],0);self.assertAlmostEqual(cyclic[1,5],1)
        self.assertLess(np.linalg.norm(cyclic[3,:2]-cyclic[4,:2]),1)
        expanded,names=calendar_representation(X,"hour_workingday69")
        self.assertEqual(expanded.shape,(5,69));self.assertEqual(names[53],"hr_08_workingday")
        self.assertEqual(expanded[1,53],1);self.assertEqual(expanded[2,45:].sum(),0)
        np.testing.assert_array_equal(expanded[:,:45],X)

    def test_direct_penalty_solve_has_small_stationarity_and_repeated_data_invariance(self):
        X=calendars();y=np.array([5,154,90,15,20.])
        model,receipt=fit_objective_model(X,y,solver="least_squares",loss=LossConfig(l2=.1))
        self.assertLess(receipt['history'][-1]['gradient_norm'],1e-9)
        repeated,_=fit_objective_model(np.vstack((X,X)),np.tile(y,2),solver="least_squares",loss=LossConfig(l2=.1))
        np.testing.assert_allclose(model.predict(X),repeated.predict(X),atol=1e-10)
        strong,_=fit_objective_model(X,y,solver="least_squares",loss=LossConfig(l2=1e8))
        np.testing.assert_allclose(strong.predict(X),np.full(5,y.mean()),atol=1e-4)

    def test_training_scaler_and_same_supported_function_in_raw_coordinates(self):
        X=calendars();y=np.array([5,154,90,15,20.])
        scaled,_=fit_objective_model(X,y,solver="least_squares")
        raw,_=fit_objective_model(X,y,solver="least_squares",scaling="raw")
        np.testing.assert_allclose(raw.predict(X),scaled.predict(X),atol=1e-10)
        self.assertTrue(scaled.standardizer.constant_columns)
        self.assertTrue(all(scaled.standardizer.scales[j]==1 for j in scaled.standardizer.constant_columns))
        before=scaled.standardizer
        scaled.predict(X[[1,1,1]])
        self.assertEqual(before,scaled.standardizer)

    def test_bounded_optimizer_and_failed_rate_have_explicit_stops(self):
        X=calendars();y=np.array([5,154,90,15,20.])
        _,r=fit_objective_model(X,y,solver="sgd",rate=.01,max_steps=2)
        self.assertEqual(r['updates'],2);self.assertEqual(r['processed_examples'],10)
        self.assertEqual(r['stop_reason'],'maximum_updates');self.assertIsNone(r['converged'])
        _,bad=fit_objective_model(X,y,solver="sgd",rate=1e100,max_steps=10)
        self.assertEqual(bad['status'],'failed');self.assertIn(bad['stop_reason'],['diverged','nonfinite'])

    def test_nonnegative_policy_is_explicit_and_does_not_mutate_input(self):
        raw=np.array([-70.,3.,9.]);truth=np.array([5.,0.,12.])
        clipped=apply_prediction_policy(raw,"nonnegative")
        np.testing.assert_array_equal(clipped,[0,3,9]);np.testing.assert_array_equal(raw,[-70,3,9])
        self.assertTrue(np.all(np.abs(clipped-truth)<=np.abs(raw-truth)))
        self.assertTrue(np.all((clipped-truth)**2<=(raw-truth)**2))

    def test_invalid_contracts_fail_before_arithmetic(self):
        for config in [dict(kind='unknown'),dict(delta=0),dict(l2=-1),dict(l2=float('nan'))]:
            with self.assertRaises(ValueError):LossConfig(**config)
        for values in [[float('nan')],[],[[1,2]]]:
            with self.assertRaises(ValueError):loss_values_and_slopes(values,LossConfig())
        with self.assertRaises(ValueError):calendar_representation(np.zeros((1,45)))
        with self.assertRaises(ValueError):apply_prediction_policy([1],'round')
        with self.assertRaises(ValueError):fit_objective_model(calendars(),[1,2,3,4,5],solver='least_squares',loss=LossConfig('mae'))


if __name__=="__main__":unittest.main()
