; ModuleID = 'DRB112-linear-orig-no.c'
source_filename = "DRB112-linear-orig-no.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@1 = private unnamed_addr constant [37 x i8] c";DRB112-linear-orig-no.c;main;66;1;;\00", align 1
@2 = private unnamed_addr constant [38 x i8] c";DRB112-linear-orig-no.c;main;66;35;;\00", align 1
@3 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@.str.1 = private unnamed_addr constant [10 x i8] c"c[50]=%f\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !9 {
entry:
  %retval = alloca i32, align 4
  %len = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  %__vla_expr1 = alloca i64, align 8
  %__vla_expr2 = alloca i64, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  %0 = bitcast %struct.ident_t* %.kmpc_loc.addr to i8*
  %1 = bitcast %struct.ident_t* @3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 24, i1 false)
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i32* %len, metadata !13, metadata !DIExpression()), !dbg !14
  store i32 100, i32* %len, align 4, !dbg !14
  %2 = load i32, i32* %len, align 4, !dbg !15
  %3 = zext i32 %2 to i64, !dbg !16
  %4 = call i8* @llvm.stacksave(), !dbg !16
  store i8* %4, i8** %saved_stack, align 8, !dbg !16
  %vla = alloca double, i64 %3, align 16, !dbg !16
  store i64 %3, i64* %__vla_expr0, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata i64* %__vla_expr0, metadata !17, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.declare(metadata double* %vla, metadata !20, metadata !DIExpression()), !dbg !24
  %5 = load i32, i32* %len, align 4, !dbg !25
  %6 = zext i32 %5 to i64, !dbg !16
  %vla1 = alloca double, i64 %6, align 16, !dbg !16
  store i64 %6, i64* %__vla_expr1, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata i64* %__vla_expr1, metadata !26, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.declare(metadata double* %vla1, metadata !27, metadata !DIExpression()), !dbg !31
  %7 = load i32, i32* %len, align 4, !dbg !32
  %8 = zext i32 %7 to i64, !dbg !16
  %vla2 = alloca double, i64 %8, align 16, !dbg !16
  store i64 %8, i64* %__vla_expr2, align 8, !dbg !16
  call void @llvm.dbg.declare(metadata i64* %__vla_expr2, metadata !33, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.declare(metadata double* %vla2, metadata !34, metadata !DIExpression()), !dbg !38
  call void @llvm.dbg.declare(metadata i32* %i, metadata !39, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata i32* %j, metadata !41, metadata !DIExpression()), !dbg !42
  store i32 0, i32* %j, align 4, !dbg !42
  store i32 0, i32* %i, align 4, !dbg !43
  br label %for.cond, !dbg !45

for.cond:                                         ; preds = %for.inc, %entry
  %9 = load i32, i32* %i, align 4, !dbg !46
  %10 = load i32, i32* %len, align 4, !dbg !48
  %cmp = icmp slt i32 %9, %10, !dbg !49
  br i1 %cmp, label %for.body, label %for.end, !dbg !50

for.body:                                         ; preds = %for.cond
  %11 = load i32, i32* %i, align 4, !dbg !51
  %conv = sitofp i32 %11 to double, !dbg !53
  %div = fdiv double %conv, 2.000000e+00, !dbg !54
  %12 = load i32, i32* %i, align 4, !dbg !55
  %idxprom = sext i32 %12 to i64, !dbg !56
  %arrayidx = getelementptr inbounds double, double* %vla, i64 %idxprom, !dbg !56
  store double %div, double* %arrayidx, align 8, !dbg !57
  %13 = load i32, i32* %i, align 4, !dbg !58
  %conv3 = sitofp i32 %13 to double, !dbg !59
  %div4 = fdiv double %conv3, 3.000000e+00, !dbg !60
  %14 = load i32, i32* %i, align 4, !dbg !61
  %idxprom5 = sext i32 %14 to i64, !dbg !62
  %arrayidx6 = getelementptr inbounds double, double* %vla1, i64 %idxprom5, !dbg !62
  store double %div4, double* %arrayidx6, align 8, !dbg !63
  %15 = load i32, i32* %i, align 4, !dbg !64
  %conv7 = sitofp i32 %15 to double, !dbg !65
  %div8 = fdiv double %conv7, 7.000000e+00, !dbg !66
  %16 = load i32, i32* %i, align 4, !dbg !67
  %idxprom9 = sext i32 %16 to i64, !dbg !68
  %arrayidx10 = getelementptr inbounds double, double* %vla2, i64 %idxprom9, !dbg !68
  store double %div8, double* %arrayidx10, align 8, !dbg !69
  br label %for.inc, !dbg !70

for.inc:                                          ; preds = %for.body
  %17 = load i32, i32* %i, align 4, !dbg !71
  %inc = add nsw i32 %17, 1, !dbg !71
  store i32 %inc, i32* %i, align 4, !dbg !71
  br label %for.cond, !dbg !72, !llvm.loop !73

for.end:                                          ; preds = %for.cond
  %18 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !75
  store i8* getelementptr inbounds ([37 x i8], [37 x i8]* @1, i32 0, i32 0), i8** %18, align 8, !dbg !75
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* %.kmpc_loc.addr, i32 8, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i64, double*, i32*, i64, double*, i64, double*)* @.omp_outlined. to void (i32*, i32*, ...)*), i32* %len, i64 %8, double* %vla2, i32* %j, i64 %3, double* %vla, i64 %6, double* %vla1), !dbg !75
  %arrayidx11 = getelementptr inbounds double, double* %vla2, i64 50, !dbg !76
  %19 = load double, double* %arrayidx11, align 16, !dbg !76
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.1, i64 0, i64 0), double %19), !dbg !77
  store i32 0, i32* %retval, align 4, !dbg !78
  %20 = load i8*, i8** %saved_stack, align 8, !dbg !79
  call void @llvm.stackrestore(i8* %20), !dbg !79
  %21 = load i32, i32* %retval, align 4, !dbg !79
  ret i32 %21, !dbg !79
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #2

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @.omp_outlined._debug__(i32* noalias %.global_tid., i32* noalias %.bound_tid., i32* dereferenceable(4) %len, i64 %vla, double* dereferenceable(8) %c, i32* dereferenceable(4) %j, i64 %vla1, double* dereferenceable(8) %a, i64 %vla3, double* dereferenceable(8) %b) #3 !dbg !80 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %len.addr = alloca i32*, align 8
  %vla.addr = alloca i64, align 8
  %c.addr = alloca double*, align 8
  %j.addr = alloca i32*, align 8
  %vla.addr2 = alloca i64, align 8
  %a.addr = alloca double*, align 8
  %vla.addr4 = alloca i64, align 8
  %b.addr = alloca double*, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.capture_expr. = alloca i32, align 4
  %.capture_expr.5 = alloca i32, align 4
  %i = alloca i32, align 4
  %.linear.start = alloca i32, align 4
  %.omp.lb = alloca i32, align 4
  %.omp.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  %i8 = alloca i32, align 4
  %j9 = alloca i32, align 4
  %0 = bitcast %struct.ident_t* %.kmpc_loc.addr to i8*
  %1 = bitcast %struct.ident_t* @0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 24, i1 false)
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !89, metadata !DIExpression()), !dbg !90
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !91, metadata !DIExpression()), !dbg !90
  store i32* %len, i32** %len.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %len.addr, metadata !92, metadata !DIExpression()), !dbg !93
  store i64 %vla, i64* %vla.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %vla.addr, metadata !94, metadata !DIExpression()), !dbg !90
  store double* %c, double** %c.addr, align 8
  call void @llvm.dbg.declare(metadata double** %c.addr, metadata !95, metadata !DIExpression()), !dbg !96
  store i32* %j, i32** %j.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %j.addr, metadata !97, metadata !DIExpression()), !dbg !98
  store i64 %vla1, i64* %vla.addr2, align 8
  call void @llvm.dbg.declare(metadata i64* %vla.addr2, metadata !99, metadata !DIExpression()), !dbg !90
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !100, metadata !DIExpression()), !dbg !101
  store i64 %vla3, i64* %vla.addr4, align 8
  call void @llvm.dbg.declare(metadata i64* %vla.addr4, metadata !102, metadata !DIExpression()), !dbg !90
  store double* %b, double** %b.addr, align 8
  call void @llvm.dbg.declare(metadata double** %b.addr, metadata !103, metadata !DIExpression()), !dbg !104
  %2 = load i32*, i32** %len.addr, align 8, !dbg !105
  %3 = load i64, i64* %vla.addr, align 8, !dbg !105
  %4 = load double*, double** %c.addr, align 8, !dbg !105
  %5 = load i32*, i32** %j.addr, align 8, !dbg !105
  %6 = load i64, i64* %vla.addr2, align 8, !dbg !105
  %7 = load double*, double** %a.addr, align 8, !dbg !105
  %8 = load i64, i64* %vla.addr4, align 8, !dbg !105
  %9 = load double*, double** %b.addr, align 8, !dbg !105
  call void @llvm.dbg.declare(metadata i32* %.omp.iv, metadata !106, metadata !DIExpression()), !dbg !90
  call void @llvm.dbg.declare(metadata i32* %.capture_expr., metadata !107, metadata !DIExpression()), !dbg !90
  %10 = load i32, i32* %2, align 4, !dbg !108
  store i32 %10, i32* %.capture_expr., align 4, !dbg !108
  call void @llvm.dbg.declare(metadata i32* %.capture_expr.5, metadata !107, metadata !DIExpression()), !dbg !90
  %11 = load i32, i32* %.capture_expr., align 4, !dbg !108
  %sub = sub nsw i32 %11, 0, !dbg !105
  %sub6 = sub nsw i32 %sub, 1, !dbg !105
  %add = add nsw i32 %sub6, 1, !dbg !105
  %div = sdiv i32 %add, 1, !dbg !105
  %sub7 = sub nsw i32 %div, 1, !dbg !105
  store i32 %sub7, i32* %.capture_expr.5, align 4, !dbg !105
  call void @llvm.dbg.declare(metadata i32* %i, metadata !109, metadata !DIExpression()), !dbg !90
  store i32 0, i32* %i, align 4, !dbg !110
  %12 = load i32, i32* %.capture_expr., align 4, !dbg !108
  %cmp = icmp slt i32 0, %12, !dbg !105
  br i1 %cmp, label %omp.precond.then, label %omp.precond.end, !dbg !105

omp.precond.then:                                 ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %.linear.start, metadata !111, metadata !DIExpression()), !dbg !90
  %13 = load i32, i32* %5, align 4, !dbg !112
  store i32 %13, i32* %.linear.start, align 4, !dbg !105
  call void @llvm.dbg.declare(metadata i32* %.omp.lb, metadata !113, metadata !DIExpression()), !dbg !90
  store i32 0, i32* %.omp.lb, align 4, !dbg !114
  call void @llvm.dbg.declare(metadata i32* %.omp.ub, metadata !115, metadata !DIExpression()), !dbg !90
  %14 = load i32, i32* %.capture_expr.5, align 4, !dbg !105
  store i32 %14, i32* %.omp.ub, align 4, !dbg !114
  call void @llvm.dbg.declare(metadata i32* %.omp.stride, metadata !116, metadata !DIExpression()), !dbg !90
  store i32 1, i32* %.omp.stride, align 4, !dbg !114
  call void @llvm.dbg.declare(metadata i32* %.omp.is_last, metadata !117, metadata !DIExpression()), !dbg !90
  store i32 0, i32* %.omp.is_last, align 4, !dbg !114
  %15 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !105
  store i8* getelementptr inbounds ([37 x i8], [37 x i8]* @1, i32 0, i32 0), i8** %15, align 8, !dbg !105
  %16 = load i32*, i32** %.global_tid..addr, align 8, !dbg !105
  %17 = load i32, i32* %16, align 4, !dbg !105
  call void @__kmpc_barrier(%struct.ident_t* %.kmpc_loc.addr, i32 %17), !dbg !105
  call void @llvm.dbg.declare(metadata i32* %i8, metadata !109, metadata !DIExpression()), !dbg !90
  call void @llvm.dbg.declare(metadata i32* %j9, metadata !118, metadata !DIExpression()), !dbg !90
  %18 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !105
  store i8* getelementptr inbounds ([37 x i8], [37 x i8]* @1, i32 0, i32 0), i8** %18, align 8, !dbg !105
  %19 = load i32*, i32** %.global_tid..addr, align 8, !dbg !105
  %20 = load i32, i32* %19, align 4, !dbg !105
  call void @__kmpc_for_static_init_4(%struct.ident_t* %.kmpc_loc.addr, i32 %20, i32 34, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1), !dbg !105
  %21 = load i32, i32* %.omp.ub, align 4, !dbg !114
  %22 = load i32, i32* %.capture_expr.5, align 4, !dbg !105
  %cmp10 = icmp sgt i32 %21, %22, !dbg !114
  br i1 %cmp10, label %cond.true, label %cond.false, !dbg !114

cond.true:                                        ; preds = %omp.precond.then
  %23 = load i32, i32* %.capture_expr.5, align 4, !dbg !105
  br label %cond.end, !dbg !114

cond.false:                                       ; preds = %omp.precond.then
  %24 = load i32, i32* %.omp.ub, align 4, !dbg !114
  br label %cond.end, !dbg !114

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %23, %cond.true ], [ %24, %cond.false ], !dbg !114
  store i32 %cond, i32* %.omp.ub, align 4, !dbg !114
  %25 = load i32, i32* %.omp.lb, align 4, !dbg !114
  store i32 %25, i32* %.omp.iv, align 4, !dbg !114
  br label %omp.inner.for.cond, !dbg !105

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc, %cond.end
  %26 = load i32, i32* %.omp.iv, align 4, !dbg !114
  %27 = load i32, i32* %.omp.ub, align 4, !dbg !114
  %cmp11 = icmp sle i32 %26, %27, !dbg !105
  br i1 %cmp11, label %omp.inner.for.body, label %omp.inner.for.end, !dbg !105

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %28 = load i32, i32* %.omp.iv, align 4, !dbg !114
  %mul = mul nsw i32 %28, 1, !dbg !110
  %add12 = add nsw i32 0, %mul, !dbg !110
  store i32 %add12, i32* %i8, align 4, !dbg !110
  %29 = load i32, i32* %.linear.start, align 4, !dbg !112
  %30 = load i32, i32* %.omp.iv, align 4, !dbg !114
  %mul13 = mul nsw i32 %30, 1, !dbg !112
  %add14 = add nsw i32 %29, %mul13, !dbg !112
  store i32 %add14, i32* %j9, align 4, !dbg !112
  %31 = load i32, i32* %i8, align 4, !dbg !119
  %idxprom = sext i32 %31 to i64, !dbg !121
  %arrayidx = getelementptr inbounds double, double* %7, i64 %idxprom, !dbg !121
  %32 = load double, double* %arrayidx, align 8, !dbg !121
  %33 = load i32, i32* %i8, align 4, !dbg !122
  %idxprom15 = sext i32 %33 to i64, !dbg !123
  %arrayidx16 = getelementptr inbounds double, double* %9, i64 %idxprom15, !dbg !123
  %34 = load double, double* %arrayidx16, align 8, !dbg !123
  %mul17 = fmul double %32, %34, !dbg !124
  %35 = load i32, i32* %j9, align 4, !dbg !125
  %idxprom18 = sext i32 %35 to i64, !dbg !126
  %arrayidx19 = getelementptr inbounds double, double* %4, i64 %idxprom18, !dbg !126
  %36 = load double, double* %arrayidx19, align 8, !dbg !127
  %add20 = fadd double %36, %mul17, !dbg !127
  store double %add20, double* %arrayidx19, align 8, !dbg !127
  %37 = load i32, i32* %j9, align 4, !dbg !128
  %inc = add nsw i32 %37, 1, !dbg !128
  store i32 %inc, i32* %j9, align 4, !dbg !128
  br label %omp.body.continue, !dbg !129

omp.body.continue:                                ; preds = %omp.inner.for.body
  br label %omp.inner.for.inc, !dbg !130

omp.inner.for.inc:                                ; preds = %omp.body.continue
  %38 = load i32, i32* %.omp.iv, align 4, !dbg !114
  %add21 = add nsw i32 %38, 1, !dbg !105
  store i32 %add21, i32* %.omp.iv, align 4, !dbg !105
  br label %omp.inner.for.cond, !dbg !130, !llvm.loop !131

omp.inner.for.end:                                ; preds = %omp.inner.for.cond
  br label %omp.loop.exit, !dbg !130

omp.loop.exit:                                    ; preds = %omp.inner.for.end
  %39 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !130
  store i8* getelementptr inbounds ([38 x i8], [38 x i8]* @2, i32 0, i32 0), i8** %39, align 8, !dbg !130
  %40 = load i32*, i32** %.global_tid..addr, align 8, !dbg !130
  %41 = load i32, i32* %40, align 4, !dbg !130
  call void @__kmpc_for_static_fini(%struct.ident_t* %.kmpc_loc.addr, i32 %41), !dbg !130
  %42 = load i32, i32* %.omp.is_last, align 4, !dbg !130
  %43 = icmp ne i32 %42, 0, !dbg !130
  br i1 %43, label %.omp.linear.pu, label %.omp.linear.pu.done, !dbg !130

.omp.linear.pu:                                   ; preds = %omp.loop.exit
  %44 = load i32, i32* %.linear.start, align 4, !dbg !112
  %45 = load i32, i32* %.capture_expr.5, align 4, !dbg !105
  %add22 = add nsw i32 %45, 1, !dbg !105
  %mul23 = mul nsw i32 %add22, 1, !dbg !112
  %add24 = add nsw i32 %44, %mul23, !dbg !112
  store i32 %add24, i32* %5, align 4, !dbg !112
  br label %.omp.linear.pu.done, !dbg !130

.omp.linear.pu.done:                              ; preds = %.omp.linear.pu, %omp.loop.exit
  br label %omp.precond.end, !dbg !130

omp.precond.end:                                  ; preds = %.omp.linear.pu.done, %entry
  ret void, !dbg !133
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #4

declare dso_local void @__kmpc_barrier(%struct.ident_t*, i32)

declare dso_local void @__kmpc_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32)

declare dso_local void @__kmpc_for_static_fini(%struct.ident_t*, i32)

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @.omp_outlined.(i32* noalias %.global_tid., i32* noalias %.bound_tid., i32* dereferenceable(4) %len, i64 %vla, double* dereferenceable(8) %c, i32* dereferenceable(4) %j, i64 %vla1, double* dereferenceable(8) %a, i64 %vla3, double* dereferenceable(8) %b) #3 !dbg !134 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %len.addr = alloca i32*, align 8
  %vla.addr = alloca i64, align 8
  %c.addr = alloca double*, align 8
  %j.addr = alloca i32*, align 8
  %vla.addr2 = alloca i64, align 8
  %a.addr = alloca double*, align 8
  %vla.addr4 = alloca i64, align 8
  %b.addr = alloca double*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !135, metadata !DIExpression()), !dbg !136
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !137, metadata !DIExpression()), !dbg !136
  store i32* %len, i32** %len.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %len.addr, metadata !138, metadata !DIExpression()), !dbg !136
  store i64 %vla, i64* %vla.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %vla.addr, metadata !139, metadata !DIExpression()), !dbg !136
  store double* %c, double** %c.addr, align 8
  call void @llvm.dbg.declare(metadata double** %c.addr, metadata !140, metadata !DIExpression()), !dbg !136
  store i32* %j, i32** %j.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %j.addr, metadata !141, metadata !DIExpression()), !dbg !136
  store i64 %vla1, i64* %vla.addr2, align 8
  call void @llvm.dbg.declare(metadata i64* %vla.addr2, metadata !142, metadata !DIExpression()), !dbg !136
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !143, metadata !DIExpression()), !dbg !136
  store i64 %vla3, i64* %vla.addr4, align 8
  call void @llvm.dbg.declare(metadata i64* %vla.addr4, metadata !144, metadata !DIExpression()), !dbg !136
  store double* %b, double** %b.addr, align 8
  call void @llvm.dbg.declare(metadata double** %b.addr, metadata !145, metadata !DIExpression()), !dbg !136
  %0 = load i32*, i32** %len.addr, align 8, !dbg !146
  %1 = load i64, i64* %vla.addr, align 8, !dbg !146
  %2 = load double*, double** %c.addr, align 8, !dbg !146
  %3 = load i32*, i32** %j.addr, align 8, !dbg !146
  %4 = load i64, i64* %vla.addr2, align 8, !dbg !146
  %5 = load double*, double** %a.addr, align 8, !dbg !146
  %6 = load i64, i64* %vla.addr4, align 8, !dbg !146
  %7 = load double*, double** %b.addr, align 8, !dbg !146
  %8 = load i32*, i32** %.global_tid..addr, align 8, !dbg !146
  %9 = load i32*, i32** %.bound_tid..addr, align 8, !dbg !146
  %10 = load i32*, i32** %len.addr, align 8, !dbg !146
  %11 = load double*, double** %c.addr, align 8, !dbg !146
  %12 = load i32*, i32** %j.addr, align 8, !dbg !146
  %13 = load double*, double** %a.addr, align 8, !dbg !146
  %14 = load double*, double** %b.addr, align 8, !dbg !146
  call void @.omp_outlined._debug__(i32* %8, i32* %9, i32* %10, i64 %1, double* %11, i32* %12, i64 %4, double* %13, i64 %6, double* %14) #2, !dbg !146
  ret void, !dbg !146
}

declare !callback !147 dso_local void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

declare dso_local i32 @printf(i8*, ...) #5

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind }
attributes #3 = { noinline norecurse nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind willreturn }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.1 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "DRB112-linear-orig-no.c", directory: "/home/yanze/code/OpenRace/tests/data/integration/dataracebench")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!5 = !{i32 7, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{!"clang version 10.0.1 "}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 53, type: !10, scopeLine: 54, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "len", scope: !9, file: !1, line: 55, type: !12)
!14 = !DILocation(line: 55, column: 7, scope: !9)
!15 = !DILocation(line: 56, column: 12, scope: !9)
!16 = !DILocation(line: 56, column: 3, scope: !9)
!17 = !DILocalVariable(name: "__vla_expr0", scope: !9, type: !18, flags: DIFlagArtificial)
!18 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!19 = !DILocation(line: 0, scope: !9)
!20 = !DILocalVariable(name: "a", scope: !9, file: !1, line: 56, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, elements: !22)
!22 = !{!23}
!23 = !DISubrange(count: !17)
!24 = !DILocation(line: 56, column: 10, scope: !9)
!25 = !DILocation(line: 56, column: 20, scope: !9)
!26 = !DILocalVariable(name: "__vla_expr1", scope: !9, type: !18, flags: DIFlagArtificial)
!27 = !DILocalVariable(name: "b", scope: !9, file: !1, line: 56, type: !28)
!28 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, elements: !29)
!29 = !{!30}
!30 = !DISubrange(count: !26)
!31 = !DILocation(line: 56, column: 18, scope: !9)
!32 = !DILocation(line: 56, column: 28, scope: !9)
!33 = !DILocalVariable(name: "__vla_expr2", scope: !9, type: !18, flags: DIFlagArtificial)
!34 = !DILocalVariable(name: "c", scope: !9, file: !1, line: 56, type: !35)
!35 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, elements: !36)
!36 = !{!37}
!37 = !DISubrange(count: !33)
!38 = !DILocation(line: 56, column: 26, scope: !9)
!39 = !DILocalVariable(name: "i", scope: !9, file: !1, line: 57, type: !12)
!40 = !DILocation(line: 57, column: 7, scope: !9)
!41 = !DILocalVariable(name: "j", scope: !9, file: !1, line: 57, type: !12)
!42 = !DILocation(line: 57, column: 9, scope: !9)
!43 = !DILocation(line: 59, column: 9, scope: !44)
!44 = distinct !DILexicalBlock(scope: !9, file: !1, line: 59, column: 3)
!45 = !DILocation(line: 59, column: 8, scope: !44)
!46 = !DILocation(line: 59, column: 12, scope: !47)
!47 = distinct !DILexicalBlock(scope: !44, file: !1, line: 59, column: 3)
!48 = !DILocation(line: 59, column: 14, scope: !47)
!49 = !DILocation(line: 59, column: 13, scope: !47)
!50 = !DILocation(line: 59, column: 3, scope: !44)
!51 = !DILocation(line: 61, column: 19, scope: !52)
!52 = distinct !DILexicalBlock(scope: !47, file: !1, line: 60, column: 3)
!53 = !DILocation(line: 61, column: 11, scope: !52)
!54 = !DILocation(line: 61, column: 21, scope: !52)
!55 = !DILocation(line: 61, column: 7, scope: !52)
!56 = !DILocation(line: 61, column: 5, scope: !52)
!57 = !DILocation(line: 61, column: 9, scope: !52)
!58 = !DILocation(line: 62, column: 19, scope: !52)
!59 = !DILocation(line: 62, column: 11, scope: !52)
!60 = !DILocation(line: 62, column: 21, scope: !52)
!61 = !DILocation(line: 62, column: 7, scope: !52)
!62 = !DILocation(line: 62, column: 5, scope: !52)
!63 = !DILocation(line: 62, column: 9, scope: !52)
!64 = !DILocation(line: 63, column: 19, scope: !52)
!65 = !DILocation(line: 63, column: 11, scope: !52)
!66 = !DILocation(line: 63, column: 21, scope: !52)
!67 = !DILocation(line: 63, column: 7, scope: !52)
!68 = !DILocation(line: 63, column: 5, scope: !52)
!69 = !DILocation(line: 63, column: 9, scope: !52)
!70 = !DILocation(line: 64, column: 3, scope: !52)
!71 = !DILocation(line: 59, column: 19, scope: !47)
!72 = !DILocation(line: 59, column: 3, scope: !47)
!73 = distinct !{!73, !50, !74}
!74 = !DILocation(line: 64, column: 3, scope: !44)
!75 = !DILocation(line: 66, column: 1, scope: !9)
!76 = !DILocation(line: 73, column: 24, scope: !9)
!77 = !DILocation(line: 73, column: 3, scope: !9)
!78 = !DILocation(line: 74, column: 3, scope: !9)
!79 = !DILocation(line: 75, column: 1, scope: !9)
!80 = distinct !DISubprogram(name: ".omp_outlined._debug__", scope: !1, file: !1, line: 67, type: !81, scopeLine: 67, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!81 = !DISubroutineType(types: !82)
!82 = !{null, !83, !83, !87, !18, !88, !87, !18, !88, !18, !88}
!83 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !84)
!84 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !85)
!85 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !86, size: 64)
!86 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!87 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !12, size: 64)
!88 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !4, size: 64)
!89 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !80, type: !83, flags: DIFlagArtificial)
!90 = !DILocation(line: 0, scope: !80)
!91 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !80, type: !83, flags: DIFlagArtificial)
!92 = !DILocalVariable(name: "len", arg: 3, scope: !80, file: !1, line: 55, type: !87)
!93 = !DILocation(line: 55, column: 7, scope: !80)
!94 = !DILocalVariable(name: "vla", arg: 4, scope: !80, type: !18, flags: DIFlagArtificial)
!95 = !DILocalVariable(name: "c", arg: 5, scope: !80, file: !1, line: 56, type: !88)
!96 = !DILocation(line: 56, column: 26, scope: !80)
!97 = !DILocalVariable(name: "j", arg: 6, scope: !80, file: !1, line: 57, type: !87)
!98 = !DILocation(line: 57, column: 9, scope: !80)
!99 = !DILocalVariable(name: "vla", arg: 7, scope: !80, type: !18, flags: DIFlagArtificial)
!100 = !DILocalVariable(name: "a", arg: 8, scope: !80, file: !1, line: 56, type: !88)
!101 = !DILocation(line: 56, column: 10, scope: !80)
!102 = !DILocalVariable(name: "vla", arg: 9, scope: !80, type: !18, flags: DIFlagArtificial)
!103 = !DILocalVariable(name: "b", arg: 10, scope: !80, file: !1, line: 56, type: !88)
!104 = !DILocation(line: 56, column: 18, scope: !80)
!105 = !DILocation(line: 67, column: 3, scope: !80)
!106 = !DILocalVariable(name: ".omp.iv", scope: !80, type: !12, flags: DIFlagArtificial)
!107 = !DILocalVariable(name: ".capture_expr.", scope: !80, type: !12, flags: DIFlagArtificial)
!108 = !DILocation(line: 67, column: 14, scope: !80)
!109 = !DILocalVariable(name: "i", scope: !80, type: !12, flags: DIFlagArtificial)
!110 = !DILocation(line: 67, column: 18, scope: !80)
!111 = !DILocalVariable(name: ".linear.start", scope: !80, type: !12, flags: DIFlagArtificial)
!112 = !DILocation(line: 66, column: 33, scope: !80)
!113 = !DILocalVariable(name: ".omp.lb", scope: !80, type: !12, flags: DIFlagArtificial)
!114 = !DILocation(line: 67, column: 8, scope: !80)
!115 = !DILocalVariable(name: ".omp.ub", scope: !80, type: !12, flags: DIFlagArtificial)
!116 = !DILocalVariable(name: ".omp.stride", scope: !80, type: !12, flags: DIFlagArtificial)
!117 = !DILocalVariable(name: ".omp.is_last", scope: !80, type: !12, flags: DIFlagArtificial)
!118 = !DILocalVariable(name: "j", scope: !80, type: !12, flags: DIFlagArtificial)
!119 = !DILocation(line: 69, column: 13, scope: !120)
!120 = distinct !DILexicalBlock(scope: !80, file: !1, line: 68, column: 3)
!121 = !DILocation(line: 69, column: 11, scope: !120)
!122 = !DILocation(line: 69, column: 18, scope: !120)
!123 = !DILocation(line: 69, column: 16, scope: !120)
!124 = !DILocation(line: 69, column: 15, scope: !120)
!125 = !DILocation(line: 69, column: 7, scope: !120)
!126 = !DILocation(line: 69, column: 5, scope: !120)
!127 = !DILocation(line: 69, column: 9, scope: !120)
!128 = !DILocation(line: 70, column: 6, scope: !120)
!129 = !DILocation(line: 71, column: 3, scope: !120)
!130 = !DILocation(line: 66, column: 1, scope: !80)
!131 = distinct !{!131, !130, !132}
!132 = !DILocation(line: 66, column: 35, scope: !80)
!133 = !DILocation(line: 71, column: 3, scope: !80)
!134 = distinct !DISubprogram(name: ".omp_outlined.", scope: !1, file: !1, line: 67, type: !81, scopeLine: 67, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!135 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !134, type: !83, flags: DIFlagArtificial)
!136 = !DILocation(line: 0, scope: !134)
!137 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !134, type: !83, flags: DIFlagArtificial)
!138 = !DILocalVariable(name: "len", arg: 3, scope: !134, type: !87, flags: DIFlagArtificial)
!139 = !DILocalVariable(name: "vla", arg: 4, scope: !134, type: !18, flags: DIFlagArtificial)
!140 = !DILocalVariable(name: "c", arg: 5, scope: !134, type: !88, flags: DIFlagArtificial)
!141 = !DILocalVariable(name: "j", arg: 6, scope: !134, type: !87, flags: DIFlagArtificial)
!142 = !DILocalVariable(name: "vla", arg: 7, scope: !134, type: !18, flags: DIFlagArtificial)
!143 = !DILocalVariable(name: "a", arg: 8, scope: !134, type: !88, flags: DIFlagArtificial)
!144 = !DILocalVariable(name: "vla", arg: 9, scope: !134, type: !18, flags: DIFlagArtificial)
!145 = !DILocalVariable(name: "b", arg: 10, scope: !134, type: !88, flags: DIFlagArtificial)
!146 = !DILocation(line: 67, column: 3, scope: !134)
!147 = !{!148}
!148 = !{i64 2, i64 -1, i64 -1, i1 true}
