; ModuleID = 'DRB161-nolocksimd-orig-gpu-yes.c'
source_filename = "DRB161-nolocksimd-orig-gpu-yes.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@0 = private unnamed_addr global %struct.ident_t { i32 0, i32 2050, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@1 = private unnamed_addr constant [46 x i8] c";DRB161-nolocksimd-orig-gpu-yes.c;main;29;3;;\00", align 1
@2 = private unnamed_addr global %struct.ident_t { i32 0, i32 514, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@3 = private unnamed_addr constant [47 x i8] c";DRB161-nolocksimd-orig-gpu-yes.c;main;29;39;;\00", align 1
@4 = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str, i32 0, i32 0) }, align 8
@5 = private unnamed_addr constant [46 x i8] c";DRB161-nolocksimd-orig-gpu-yes.c;main;28;3;;\00", align 1
@.str.3 = private unnamed_addr constant [5 x i8] c"%d\0A \00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %var = alloca [8 x i32], align 16
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata [8 x i32]* %var, metadata !11, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32* %i, metadata !16, metadata !DIExpression()), !dbg !18
  store i32 0, i32* %i, align 4, !dbg !18
  br label %for.cond, !dbg !19

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4, !dbg !20
  %cmp = icmp slt i32 %0, 8, !dbg !22
  br i1 %cmp, label %for.body, label %for.end, !dbg !23

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4, !dbg !24
  %idxprom = sext i32 %1 to i64, !dbg !26
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %var, i64 0, i64 %idxprom, !dbg !26
  store i32 0, i32* %arrayidx, align 4, !dbg !27
  br label %for.inc, !dbg !28

for.inc:                                          ; preds = %for.body
  %2 = load i32, i32* %i, align 4, !dbg !29
  %inc = add nsw i32 %2, 1, !dbg !29
  store i32 %inc, i32* %i, align 4, !dbg !29
  br label %for.cond, !dbg !30, !llvm.loop !31

for.end:                                          ; preds = %for.cond
  call void @__omp_offloading_10307_2ec41ba_main_l27([8 x i32]* %var) #5, !dbg !33
  call void @llvm.dbg.declare(metadata i32* %i1, metadata !35, metadata !DIExpression()), !dbg !37
  store i32 0, i32* %i1, align 4, !dbg !37
  br label %for.cond2, !dbg !38

for.cond2:                                        ; preds = %for.inc10, %for.end
  %3 = load i32, i32* %i1, align 4, !dbg !39
  %cmp3 = icmp slt i32 %3, 8, !dbg !41
  br i1 %cmp3, label %for.body4, label %for.end12, !dbg !42

for.body4:                                        ; preds = %for.cond2
  %4 = load i32, i32* %i1, align 4, !dbg !43
  %idxprom5 = sext i32 %4 to i64, !dbg !46
  %arrayidx6 = getelementptr inbounds [8 x i32], [8 x i32]* %var, i64 0, i64 %idxprom5, !dbg !46
  %5 = load i32, i32* %arrayidx6, align 4, !dbg !46
  %cmp7 = icmp ne i32 %5, 20, !dbg !47
  br i1 %cmp7, label %if.then, label %if.end, !dbg !48

if.then:                                          ; preds = %for.body4
  %6 = load i32, i32* %i1, align 4, !dbg !49
  %idxprom8 = sext i32 %6 to i64, !dbg !50
  %arrayidx9 = getelementptr inbounds [8 x i32], [8 x i32]* %var, i64 0, i64 %idxprom8, !dbg !50
  %7 = load i32, i32* %arrayidx9, align 4, !dbg !50
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0), i32 %7), !dbg !51
  br label %if.end, !dbg !51

if.end:                                           ; preds = %if.then, %for.body4
  br label %for.inc10, !dbg !52

for.inc10:                                        ; preds = %if.end
  %8 = load i32, i32* %i1, align 4, !dbg !53
  %inc11 = add nsw i32 %8, 1, !dbg !53
  store i32 %inc11, i32* %i1, align 4, !dbg !53
  br label %for.cond2, !dbg !54, !llvm.loop !55

for.end12:                                        ; preds = %for.cond2
  ret i32 0, !dbg !57
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @__omp_offloading_10307_2ec41ba_main_l27_debug__([8 x i32]* dereferenceable(32) %var) #2 !dbg !58 {
entry:
  %var.addr = alloca [8 x i32]*, align 8
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  %0 = bitcast %struct.ident_t* %.kmpc_loc.addr to i8*
  %1 = bitcast %struct.ident_t* @4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 24, i1 false)
  %2 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4
  store i8* getelementptr inbounds ([46 x i8], [46 x i8]* @5, i32 0, i32 0), i8** %2, align 8
  %3 = call i32 @__kmpc_global_thread_num(%struct.ident_t* %.kmpc_loc.addr)
  store [8 x i32]* %var, [8 x i32]** %var.addr, align 8
  call void @llvm.dbg.declare(metadata [8 x i32]** %var.addr, metadata !62, metadata !DIExpression()), !dbg !63
  %4 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !64
  %5 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !64
  store i8* getelementptr inbounds ([46 x i8], [46 x i8]* @5, i32 0, i32 0), i8** %5, align 8, !dbg !64
  %6 = call i32 @__kmpc_push_num_teams(%struct.ident_t* %.kmpc_loc.addr, i32 %3, i32 1, i32 1048), !dbg !64
  %7 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !64
  store i8* getelementptr inbounds ([46 x i8], [46 x i8]* @5, i32 0, i32 0), i8** %7, align 8, !dbg !64
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* %.kmpc_loc.addr, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [8 x i32]*)* @.omp_outlined..2 to void (i32*, i32*, ...)*), [8 x i32]* %4), !dbg !64
  ret void, !dbg !65
}

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @.omp_outlined._debug__(i32* noalias %.global_tid., i32* noalias %.bound_tid., [8 x i32]* dereferenceable(32) %var) #2 !dbg !66 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %var.addr = alloca [8 x i32]*, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.omp.comb.lb = alloca i32, align 4
  %.omp.comb.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %i = alloca i32, align 4
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  %0 = bitcast %struct.ident_t* %.kmpc_loc.addr to i8*
  %1 = bitcast %struct.ident_t* @0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 24, i1 false)
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !73, metadata !DIExpression()), !dbg !74
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !75, metadata !DIExpression()), !dbg !74
  store [8 x i32]* %var, [8 x i32]** %var.addr, align 8
  call void @llvm.dbg.declare(metadata [8 x i32]** %var.addr, metadata !76, metadata !DIExpression()), !dbg !77
  %2 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !78
  call void @llvm.dbg.declare(metadata i32* %.omp.iv, metadata !79, metadata !DIExpression()), !dbg !81
  call void @llvm.dbg.declare(metadata i32* %.omp.comb.lb, metadata !82, metadata !DIExpression()), !dbg !81
  store i32 0, i32* %.omp.comb.lb, align 4, !dbg !83
  call void @llvm.dbg.declare(metadata i32* %.omp.comb.ub, metadata !84, metadata !DIExpression()), !dbg !81
  store i32 19, i32* %.omp.comb.ub, align 4, !dbg !83
  call void @llvm.dbg.declare(metadata i32* %.omp.stride, metadata !85, metadata !DIExpression()), !dbg !81
  store i32 1, i32* %.omp.stride, align 4, !dbg !83
  call void @llvm.dbg.declare(metadata i32* %.omp.is_last, metadata !86, metadata !DIExpression()), !dbg !81
  store i32 0, i32* %.omp.is_last, align 4, !dbg !83
  call void @llvm.dbg.declare(metadata i32* %i, metadata !87, metadata !DIExpression()), !dbg !81
  %3 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !78
  store i8* getelementptr inbounds ([46 x i8], [46 x i8]* @1, i32 0, i32 0), i8** %3, align 8, !dbg !78
  %4 = load i32*, i32** %.global_tid..addr, align 8, !dbg !78
  %5 = load i32, i32* %4, align 4, !dbg !78
  call void @__kmpc_for_static_init_4(%struct.ident_t* %.kmpc_loc.addr, i32 %5, i32 92, i32* %.omp.is_last, i32* %.omp.comb.lb, i32* %.omp.comb.ub, i32* %.omp.stride, i32 1, i32 1), !dbg !78
  %6 = load i32, i32* %.omp.comb.ub, align 4, !dbg !83
  %cmp = icmp sgt i32 %6, 19, !dbg !83
  br i1 %cmp, label %cond.true, label %cond.false, !dbg !83

cond.true:                                        ; preds = %entry
  br label %cond.end, !dbg !83

cond.false:                                       ; preds = %entry
  %7 = load i32, i32* %.omp.comb.ub, align 4, !dbg !83
  br label %cond.end, !dbg !83

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 19, %cond.true ], [ %7, %cond.false ], !dbg !83
  store i32 %cond, i32* %.omp.comb.ub, align 4, !dbg !83
  %8 = load i32, i32* %.omp.comb.lb, align 4, !dbg !83
  store i32 %8, i32* %.omp.iv, align 4, !dbg !83
  br label %omp.inner.for.cond, !dbg !78

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc, %cond.end
  %9 = load i32, i32* %.omp.iv, align 4, !dbg !83
  %10 = load i32, i32* %.omp.comb.ub, align 4, !dbg !83
  %cmp1 = icmp sle i32 %9, %10, !dbg !88
  br i1 %cmp1, label %omp.inner.for.body, label %omp.inner.for.end, !dbg !78

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %11 = load i32, i32* %.omp.comb.lb, align 4, !dbg !89
  %12 = zext i32 %11 to i64, !dbg !89
  %13 = load i32, i32* %.omp.comb.ub, align 4, !dbg !89
  %14 = zext i32 %13 to i64, !dbg !89
  %15 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !89
  store i8* getelementptr inbounds ([46 x i8], [46 x i8]* @1, i32 0, i32 0), i8** %15, align 8, !dbg !89
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* %.kmpc_loc.addr, i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, i64, [8 x i32]*)* @.omp_outlined. to void (i32*, i32*, ...)*), i64 %12, i64 %14, [8 x i32]* %2), !dbg !89
  br label %omp.inner.for.inc, !dbg !90

omp.inner.for.inc:                                ; preds = %omp.inner.for.body
  %16 = load i32, i32* %.omp.iv, align 4, !dbg !83
  %17 = load i32, i32* %.omp.stride, align 4, !dbg !83
  %add = add nsw i32 %16, %17, !dbg !88
  store i32 %add, i32* %.omp.iv, align 4, !dbg !88
  br label %omp.inner.for.cond, !dbg !90, !llvm.loop !92

omp.inner.for.end:                                ; preds = %omp.inner.for.cond
  br label %omp.loop.exit, !dbg !90

omp.loop.exit:                                    ; preds = %omp.inner.for.end
  %18 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !90
  store i8* getelementptr inbounds ([46 x i8], [46 x i8]* @1, i32 0, i32 0), i8** %18, align 8, !dbg !90
  call void @__kmpc_for_static_fini(%struct.ident_t* %.kmpc_loc.addr, i32 %5), !dbg !90
  ret void, !dbg !94
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #3

declare dso_local void @__kmpc_for_static_init_4(%struct.ident_t*, i32, i32, i32*, i32*, i32*, i32*, i32, i32)

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @.omp_outlined._debug__.1(i32* noalias %.global_tid., i32* noalias %.bound_tid., i64 %.previous.lb., i64 %.previous.ub., [8 x i32]* dereferenceable(32) %var) #2 !dbg !95 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %.previous.lb..addr = alloca i64, align 8
  %.previous.ub..addr = alloca i64, align 8
  %var.addr = alloca [8 x i32]*, align 8
  %.omp.iv = alloca i32, align 4
  %tmp = alloca i32, align 4
  %.omp.lb = alloca i32, align 4
  %.omp.ub = alloca i32, align 4
  %.omp.stride = alloca i32, align 4
  %.omp.is_last = alloca i32, align 4
  %i = alloca i32, align 4
  %.kmpc_loc.addr = alloca %struct.ident_t, align 8
  %tmp5 = alloca i32, align 4
  %.omp.iv6 = alloca i32, align 4
  %i7 = alloca i32, align 4
  %0 = bitcast %struct.ident_t* %.kmpc_loc.addr to i8*
  %1 = bitcast %struct.ident_t* @2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 24, i1 false)
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !100, metadata !DIExpression()), !dbg !101
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !102, metadata !DIExpression()), !dbg !101
  store i64 %.previous.lb., i64* %.previous.lb..addr, align 8
  call void @llvm.dbg.declare(metadata i64* %.previous.lb..addr, metadata !103, metadata !DIExpression()), !dbg !101
  store i64 %.previous.ub., i64* %.previous.ub..addr, align 8
  call void @llvm.dbg.declare(metadata i64* %.previous.ub..addr, metadata !104, metadata !DIExpression()), !dbg !101
  store [8 x i32]* %var, [8 x i32]** %var.addr, align 8
  call void @llvm.dbg.declare(metadata [8 x i32]** %var.addr, metadata !105, metadata !DIExpression()), !dbg !106
  %2 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !107
  call void @llvm.dbg.declare(metadata i32* %.omp.iv, metadata !108, metadata !DIExpression()), !dbg !101
  call void @llvm.dbg.declare(metadata i32* %.omp.lb, metadata !109, metadata !DIExpression()), !dbg !101
  store i32 0, i32* %.omp.lb, align 4, !dbg !110
  call void @llvm.dbg.declare(metadata i32* %.omp.ub, metadata !111, metadata !DIExpression()), !dbg !101
  store i32 19, i32* %.omp.ub, align 4, !dbg !110
  %3 = load i64, i64* %.previous.lb..addr, align 8, !dbg !107
  %conv = trunc i64 %3 to i32, !dbg !107
  %4 = load i64, i64* %.previous.ub..addr, align 8, !dbg !107
  %conv1 = trunc i64 %4 to i32, !dbg !107
  store i32 %conv, i32* %.omp.lb, align 4, !dbg !107
  store i32 %conv1, i32* %.omp.ub, align 4, !dbg !107
  call void @llvm.dbg.declare(metadata i32* %.omp.stride, metadata !112, metadata !DIExpression()), !dbg !101
  store i32 1, i32* %.omp.stride, align 4, !dbg !110
  call void @llvm.dbg.declare(metadata i32* %.omp.is_last, metadata !113, metadata !DIExpression()), !dbg !101
  store i32 0, i32* %.omp.is_last, align 4, !dbg !110
  call void @llvm.dbg.declare(metadata i32* %i, metadata !114, metadata !DIExpression()), !dbg !101
  %5 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !107
  store i8* getelementptr inbounds ([46 x i8], [46 x i8]* @1, i32 0, i32 0), i8** %5, align 8, !dbg !107
  %6 = load i32*, i32** %.global_tid..addr, align 8, !dbg !107
  %7 = load i32, i32* %6, align 4, !dbg !107
  call void @__kmpc_for_static_init_4(%struct.ident_t* %.kmpc_loc.addr, i32 %7, i32 34, i32* %.omp.is_last, i32* %.omp.lb, i32* %.omp.ub, i32* %.omp.stride, i32 1, i32 1), !dbg !107
  %8 = load i32, i32* %.omp.ub, align 4, !dbg !110
  %cmp = icmp sgt i32 %8, 19, !dbg !110
  br i1 %cmp, label %cond.true, label %cond.false, !dbg !110

cond.true:                                        ; preds = %entry
  br label %cond.end, !dbg !110

cond.false:                                       ; preds = %entry
  %9 = load i32, i32* %.omp.ub, align 4, !dbg !110
  br label %cond.end, !dbg !110

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 19, %cond.true ], [ %9, %cond.false ], !dbg !110
  store i32 %cond, i32* %.omp.ub, align 4, !dbg !110
  %10 = load i32, i32* %.omp.lb, align 4, !dbg !110
  store i32 %10, i32* %.omp.iv, align 4, !dbg !110
  br label %omp.inner.for.cond, !dbg !107

omp.inner.for.cond:                               ; preds = %omp.inner.for.inc16, %cond.end
  %11 = load i32, i32* %.omp.iv, align 4, !dbg !110
  %12 = load i32, i32* %.omp.ub, align 4, !dbg !110
  %cmp3 = icmp sle i32 %11, %12, !dbg !107
  br i1 %cmp3, label %omp.inner.for.body, label %omp.inner.for.end18, !dbg !107

omp.inner.for.body:                               ; preds = %omp.inner.for.cond
  %13 = load i32, i32* %.omp.iv, align 4, !dbg !110
  %mul = mul nsw i32 %13, 1, !dbg !115
  %add = add nsw i32 0, %mul, !dbg !115
  store i32 %add, i32* %i, align 4, !dbg !115
  call void @llvm.dbg.declare(metadata i32* %.omp.iv6, metadata !116, metadata !DIExpression()), !dbg !119
  store i32 0, i32* %.omp.iv6, align 4, !dbg !120
  call void @llvm.dbg.declare(metadata i32* %i7, metadata !121, metadata !DIExpression()), !dbg !119
  br label %omp.inner.for.cond8, !dbg !122

omp.inner.for.cond8:                              ; preds = %omp.inner.for.inc, %omp.inner.for.body
  %14 = load i32, i32* %.omp.iv6, align 4, !dbg !120, !llvm.access.group !123
  %cmp9 = icmp slt i32 %14, 8, !dbg !124
  br i1 %cmp9, label %omp.inner.for.body11, label %omp.inner.for.end, !dbg !122

omp.inner.for.body11:                             ; preds = %omp.inner.for.cond8
  %15 = load i32, i32* %.omp.iv6, align 4, !dbg !120, !llvm.access.group !123
  %mul12 = mul nsw i32 %15, 1, !dbg !125
  %add13 = add nsw i32 0, %mul12, !dbg !125
  store i32 %add13, i32* %i7, align 4, !dbg !125, !llvm.access.group !123
  %16 = load i32, i32* %i7, align 4, !dbg !126, !llvm.access.group !123
  %idxprom = sext i32 %16 to i64, !dbg !128
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %2, i64 0, i64 %idxprom, !dbg !128
  %17 = load i32, i32* %arrayidx, align 4, !dbg !129, !llvm.access.group !123
  %inc = add nsw i32 %17, 1, !dbg !129
  store i32 %inc, i32* %arrayidx, align 4, !dbg !129, !llvm.access.group !123
  br label %omp.body.continue, !dbg !130

omp.body.continue:                                ; preds = %omp.inner.for.body11
  br label %omp.inner.for.inc, !dbg !131

omp.inner.for.inc:                                ; preds = %omp.body.continue
  %18 = load i32, i32* %.omp.iv6, align 4, !dbg !120, !llvm.access.group !123
  %add14 = add nsw i32 %18, 1, !dbg !124
  store i32 %add14, i32* %.omp.iv6, align 4, !dbg !124, !llvm.access.group !123
  br label %omp.inner.for.cond8, !dbg !131, !llvm.loop !132

omp.inner.for.end:                                ; preds = %omp.inner.for.cond8
  store i32 8, i32* %i7, align 4, !dbg !125
  br label %omp.body.continue15, !dbg !136

omp.body.continue15:                              ; preds = %omp.inner.for.end
  br label %omp.inner.for.inc16, !dbg !137

omp.inner.for.inc16:                              ; preds = %omp.body.continue15
  %19 = load i32, i32* %.omp.iv, align 4, !dbg !110
  %add17 = add nsw i32 %19, 1, !dbg !107
  store i32 %add17, i32* %.omp.iv, align 4, !dbg !107
  br label %omp.inner.for.cond, !dbg !137, !llvm.loop !138

omp.inner.for.end18:                              ; preds = %omp.inner.for.cond
  br label %omp.loop.exit, !dbg !137

omp.loop.exit:                                    ; preds = %omp.inner.for.end18
  %20 = getelementptr inbounds %struct.ident_t, %struct.ident_t* %.kmpc_loc.addr, i32 0, i32 4, !dbg !137
  store i8* getelementptr inbounds ([47 x i8], [47 x i8]* @3, i32 0, i32 0), i8** %20, align 8, !dbg !137
  call void @__kmpc_for_static_fini(%struct.ident_t* %.kmpc_loc.addr, i32 %7), !dbg !137
  ret void, !dbg !140
}

declare dso_local void @__kmpc_for_static_fini(%struct.ident_t*, i32)

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @.omp_outlined.(i32* noalias %.global_tid., i32* noalias %.bound_tid., i64 %.previous.lb., i64 %.previous.ub., [8 x i32]* dereferenceable(32) %var) #2 !dbg !141 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %.previous.lb..addr = alloca i64, align 8
  %.previous.ub..addr = alloca i64, align 8
  %var.addr = alloca [8 x i32]*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !142, metadata !DIExpression()), !dbg !143
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !144, metadata !DIExpression()), !dbg !143
  store i64 %.previous.lb., i64* %.previous.lb..addr, align 8
  call void @llvm.dbg.declare(metadata i64* %.previous.lb..addr, metadata !145, metadata !DIExpression()), !dbg !143
  store i64 %.previous.ub., i64* %.previous.ub..addr, align 8
  call void @llvm.dbg.declare(metadata i64* %.previous.ub..addr, metadata !146, metadata !DIExpression()), !dbg !143
  store [8 x i32]* %var, [8 x i32]** %var.addr, align 8
  call void @llvm.dbg.declare(metadata [8 x i32]** %var.addr, metadata !147, metadata !DIExpression()), !dbg !143
  %0 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !148
  %1 = load i32*, i32** %.global_tid..addr, align 8, !dbg !148
  %2 = load i32*, i32** %.bound_tid..addr, align 8, !dbg !148
  %3 = load i64, i64* %.previous.lb..addr, align 8, !dbg !148
  %4 = load i64, i64* %.previous.ub..addr, align 8, !dbg !148
  %5 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !148
  call void @.omp_outlined._debug__.1(i32* %1, i32* %2, i64 %3, i64 %4, [8 x i32]* %5) #5, !dbg !148
  ret void, !dbg !148
}

declare !callback !149 dso_local void @__kmpc_fork_call(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @.omp_outlined..2(i32* noalias %.global_tid., i32* noalias %.bound_tid., [8 x i32]* dereferenceable(32) %var) #2 !dbg !151 {
entry:
  %.global_tid..addr = alloca i32*, align 8
  %.bound_tid..addr = alloca i32*, align 8
  %var.addr = alloca [8 x i32]*, align 8
  store i32* %.global_tid., i32** %.global_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.global_tid..addr, metadata !152, metadata !DIExpression()), !dbg !153
  store i32* %.bound_tid., i32** %.bound_tid..addr, align 8
  call void @llvm.dbg.declare(metadata i32** %.bound_tid..addr, metadata !154, metadata !DIExpression()), !dbg !153
  store [8 x i32]* %var, [8 x i32]** %var.addr, align 8
  call void @llvm.dbg.declare(metadata [8 x i32]** %var.addr, metadata !155, metadata !DIExpression()), !dbg !153
  %0 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !156
  %1 = load i32*, i32** %.global_tid..addr, align 8, !dbg !156
  %2 = load i32*, i32** %.bound_tid..addr, align 8, !dbg !156
  %3 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !156
  call void @.omp_outlined._debug__(i32* %1, i32* %2, [8 x i32]* %3) #5, !dbg !156
  ret void, !dbg !156
}

declare dso_local i32 @__kmpc_global_thread_num(%struct.ident_t*)

declare dso_local i32 @__kmpc_push_num_teams(%struct.ident_t*, i32, i32, i32)

declare !callback !149 dso_local void @__kmpc_fork_teams(%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...)

; Function Attrs: noinline norecurse nounwind optnone uwtable
define internal void @__omp_offloading_10307_2ec41ba_main_l27([8 x i32]* dereferenceable(32) %var) #2 !dbg !157 {
entry:
  %var.addr = alloca [8 x i32]*, align 8
  store [8 x i32]* %var, [8 x i32]** %var.addr, align 8
  call void @llvm.dbg.declare(metadata [8 x i32]** %var.addr, metadata !158, metadata !DIExpression()), !dbg !159
  %0 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !160
  %1 = load [8 x i32]*, [8 x i32]** %var.addr, align 8, !dbg !160
  call void @__omp_offloading_10307_2ec41ba_main_l27_debug__([8 x i32]* %1) #5, !dbg !160
  ret void, !dbg !160
}

declare dso_local i32 @printf(i8*, ...) #4

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { noinline norecurse nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind willreturn }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.1 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "DRB161-nolocksimd-orig-gpu-yes.c", directory: "/home/yanze/code/OpenRace/tests/data/integration/dataracebench")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.1 "}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 20, type: !8, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "var", scope: !7, file: !1, line: 21, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 256, elements: !13)
!13 = !{!14}
!14 = !DISubrange(count: 8)
!15 = !DILocation(line: 21, column: 7, scope: !7)
!16 = !DILocalVariable(name: "i", scope: !17, file: !1, line: 23, type: !10)
!17 = distinct !DILexicalBlock(scope: !7, file: !1, line: 23, column: 3)
!18 = !DILocation(line: 23, column: 11, scope: !17)
!19 = !DILocation(line: 23, column: 7, scope: !17)
!20 = !DILocation(line: 23, column: 16, scope: !21)
!21 = distinct !DILexicalBlock(scope: !17, file: !1, line: 23, column: 3)
!22 = !DILocation(line: 23, column: 17, scope: !21)
!23 = !DILocation(line: 23, column: 3, scope: !17)
!24 = !DILocation(line: 24, column: 9, scope: !25)
!25 = distinct !DILexicalBlock(scope: !21, file: !1, line: 23, column: 25)
!26 = !DILocation(line: 24, column: 5, scope: !25)
!27 = !DILocation(line: 24, column: 12, scope: !25)
!28 = !DILocation(line: 25, column: 3, scope: !25)
!29 = !DILocation(line: 23, column: 22, scope: !21)
!30 = !DILocation(line: 23, column: 3, scope: !21)
!31 = distinct !{!31, !23, !32}
!32 = !DILocation(line: 25, column: 3, scope: !17)
!33 = !DILocation(line: 27, column: 3, scope: !34)
!34 = distinct !DILexicalBlock(scope: !7, file: !1, line: 27, column: 3)
!35 = !DILocalVariable(name: "i", scope: !36, file: !1, line: 37, type: !10)
!36 = distinct !DILexicalBlock(scope: !7, file: !1, line: 37, column: 3)
!37 = !DILocation(line: 37, column: 11, scope: !36)
!38 = !DILocation(line: 37, column: 7, scope: !36)
!39 = !DILocation(line: 37, column: 16, scope: !40)
!40 = distinct !DILexicalBlock(scope: !36, file: !1, line: 37, column: 3)
!41 = !DILocation(line: 37, column: 17, scope: !40)
!42 = !DILocation(line: 37, column: 3, scope: !36)
!43 = !DILocation(line: 38, column: 12, scope: !44)
!44 = distinct !DILexicalBlock(scope: !45, file: !1, line: 38, column: 8)
!45 = distinct !DILexicalBlock(scope: !40, file: !1, line: 37, column: 25)
!46 = !DILocation(line: 38, column: 8, scope: !44)
!47 = !DILocation(line: 38, column: 14, scope: !44)
!48 = !DILocation(line: 38, column: 8, scope: !45)
!49 = !DILocation(line: 38, column: 38, scope: !44)
!50 = !DILocation(line: 38, column: 34, scope: !44)
!51 = !DILocation(line: 38, column: 19, scope: !44)
!52 = !DILocation(line: 39, column: 3, scope: !45)
!53 = !DILocation(line: 37, column: 22, scope: !40)
!54 = !DILocation(line: 37, column: 3, scope: !40)
!55 = distinct !{!55, !42, !56}
!56 = !DILocation(line: 39, column: 3, scope: !36)
!57 = !DILocation(line: 41, column: 3, scope: !7)
!58 = distinct !DISubprogram(name: "__omp_offloading_10307_2ec41ba_main_l27_debug__", scope: !1, file: !1, line: 28, type: !59, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!59 = !DISubroutineType(types: !60)
!60 = !{null, !61}
!61 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !12, size: 64)
!62 = !DILocalVariable(name: "var", arg: 1, scope: !58, file: !1, line: 21, type: !61)
!63 = !DILocation(line: 21, column: 7, scope: !58)
!64 = !DILocation(line: 28, column: 3, scope: !58)
!65 = !DILocation(line: 28, column: 53, scope: !58)
!66 = distinct !DISubprogram(name: ".omp_outlined._debug__", scope: !1, file: !1, line: 29, type: !67, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!67 = !DISubroutineType(types: !68)
!68 = !{null, !69, !69, !61}
!69 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !70)
!70 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !71)
!71 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !72, size: 64)
!72 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!73 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !66, type: !69, flags: DIFlagArtificial)
!74 = !DILocation(line: 0, scope: !66)
!75 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !66, type: !69, flags: DIFlagArtificial)
!76 = !DILocalVariable(name: "var", arg: 3, scope: !66, file: !1, line: 21, type: !61)
!77 = !DILocation(line: 21, column: 7, scope: !66)
!78 = !DILocation(line: 29, column: 3, scope: !66)
!79 = !DILocalVariable(name: ".omp.iv", scope: !80, type: !10, flags: DIFlagArtificial)
!80 = distinct !DILexicalBlock(scope: !66, file: !1, line: 29, column: 3)
!81 = !DILocation(line: 0, scope: !80)
!82 = !DILocalVariable(name: ".omp.comb.lb", scope: !80, type: !10, flags: DIFlagArtificial)
!83 = !DILocation(line: 30, column: 8, scope: !80)
!84 = !DILocalVariable(name: ".omp.comb.ub", scope: !80, type: !10, flags: DIFlagArtificial)
!85 = !DILocalVariable(name: ".omp.stride", scope: !80, type: !10, flags: DIFlagArtificial)
!86 = !DILocalVariable(name: ".omp.is_last", scope: !80, type: !10, flags: DIFlagArtificial)
!87 = !DILocalVariable(name: "i", scope: !80, type: !10, flags: DIFlagArtificial)
!88 = !DILocation(line: 30, column: 3, scope: !80)
!89 = !DILocation(line: 29, column: 3, scope: !80)
!90 = !DILocation(line: 29, column: 39, scope: !91)
!91 = distinct !DILexicalBlock(scope: !80, file: !1, line: 29, column: 3)
!92 = distinct !{!92, !89, !93}
!93 = !DILocation(line: 29, column: 39, scope: !80)
!94 = !DILocation(line: 29, column: 39, scope: !66)
!95 = distinct !DISubprogram(name: ".omp_outlined._debug__.1", scope: !1, file: !1, line: 30, type: !96, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!96 = !DISubroutineType(types: !97)
!97 = !{null, !69, !69, !98, !98, !61}
!98 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !99)
!99 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!100 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !95, type: !69, flags: DIFlagArtificial)
!101 = !DILocation(line: 0, scope: !95)
!102 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !95, type: !69, flags: DIFlagArtificial)
!103 = !DILocalVariable(name: ".previous.lb.", arg: 3, scope: !95, type: !98, flags: DIFlagArtificial)
!104 = !DILocalVariable(name: ".previous.ub.", arg: 4, scope: !95, type: !98, flags: DIFlagArtificial)
!105 = !DILocalVariable(name: "var", arg: 5, scope: !95, file: !1, line: 21, type: !61)
!106 = !DILocation(line: 21, column: 7, scope: !95)
!107 = !DILocation(line: 30, column: 3, scope: !95)
!108 = !DILocalVariable(name: ".omp.iv", scope: !95, type: !10, flags: DIFlagArtificial)
!109 = !DILocalVariable(name: ".omp.lb", scope: !95, type: !10, flags: DIFlagArtificial)
!110 = !DILocation(line: 30, column: 8, scope: !95)
!111 = !DILocalVariable(name: ".omp.ub", scope: !95, type: !10, flags: DIFlagArtificial)
!112 = !DILocalVariable(name: ".omp.stride", scope: !95, type: !10, flags: DIFlagArtificial)
!113 = !DILocalVariable(name: ".omp.is_last", scope: !95, type: !10, flags: DIFlagArtificial)
!114 = !DILocalVariable(name: "i", scope: !95, type: !10, flags: DIFlagArtificial)
!115 = !DILocation(line: 30, column: 22, scope: !95)
!116 = !DILocalVariable(name: ".omp.iv", scope: !117, type: !10, flags: DIFlagArtificial)
!117 = distinct !DILexicalBlock(scope: !118, file: !1, line: 31, column: 5)
!118 = distinct !DILexicalBlock(scope: !95, file: !1, line: 30, column: 26)
!119 = !DILocation(line: 0, scope: !117)
!120 = !DILocation(line: 32, column: 9, scope: !117)
!121 = !DILocalVariable(name: "i", scope: !117, type: !10, flags: DIFlagArtificial)
!122 = !DILocation(line: 31, column: 5, scope: !118)
!123 = distinct !{}
!124 = !DILocation(line: 32, column: 5, scope: !117)
!125 = !DILocation(line: 32, column: 23, scope: !117)
!126 = !DILocation(line: 33, column: 11, scope: !127)
!127 = distinct !DILexicalBlock(scope: !117, file: !1, line: 32, column: 27)
!128 = !DILocation(line: 33, column: 7, scope: !127)
!129 = !DILocation(line: 33, column: 13, scope: !127)
!130 = !DILocation(line: 34, column: 5, scope: !127)
!131 = !DILocation(line: 31, column: 5, scope: !117)
!132 = distinct !{!132, !131, !133, !134, !135}
!133 = !DILocation(line: 31, column: 21, scope: !117)
!134 = !{!"llvm.loop.parallel_accesses", !123}
!135 = !{!"llvm.loop.vectorize.enable", i1 true}
!136 = !DILocation(line: 35, column: 3, scope: !118)
!137 = !DILocation(line: 29, column: 3, scope: !95)
!138 = distinct !{!138, !137, !139}
!139 = !DILocation(line: 29, column: 39, scope: !95)
!140 = !DILocation(line: 35, column: 3, scope: !95)
!141 = distinct !DISubprogram(name: ".omp_outlined.", scope: !1, file: !1, line: 30, type: !96, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!142 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !141, type: !69, flags: DIFlagArtificial)
!143 = !DILocation(line: 0, scope: !141)
!144 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !141, type: !69, flags: DIFlagArtificial)
!145 = !DILocalVariable(name: ".previous.lb.", arg: 3, scope: !141, type: !98, flags: DIFlagArtificial)
!146 = !DILocalVariable(name: ".previous.ub.", arg: 4, scope: !141, type: !98, flags: DIFlagArtificial)
!147 = !DILocalVariable(name: "var", arg: 5, scope: !141, type: !61, flags: DIFlagArtificial)
!148 = !DILocation(line: 30, column: 3, scope: !141)
!149 = !{!150}
!150 = !{i64 2, i64 -1, i64 -1, i1 true}
!151 = distinct !DISubprogram(name: ".omp_outlined..2", scope: !1, file: !1, line: 29, type: !67, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!152 = !DILocalVariable(name: ".global_tid.", arg: 1, scope: !151, type: !69, flags: DIFlagArtificial)
!153 = !DILocation(line: 0, scope: !151)
!154 = !DILocalVariable(name: ".bound_tid.", arg: 2, scope: !151, type: !69, flags: DIFlagArtificial)
!155 = !DILocalVariable(name: "var", arg: 3, scope: !151, type: !61, flags: DIFlagArtificial)
!156 = !DILocation(line: 29, column: 3, scope: !151)
!157 = distinct !DISubprogram(name: "__omp_offloading_10307_2ec41ba_main_l27", scope: !1, file: !1, line: 28, type: !59, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
!158 = !DILocalVariable(name: "var", arg: 1, scope: !157, type: !61, flags: DIFlagArtificial)
!159 = !DILocation(line: 0, scope: !157)
!160 = !DILocation(line: 28, column: 3, scope: !157)
