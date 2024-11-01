# Create submission file
fin_pred = pd.DataFrame()
fin_pred['id'] = sample['id']
fin_pred['loan_status'] = test_pred_cat
fin_pred.to_csv('submission.csv', index=False) 